"""Same-band regional products from a converged multi-map detection stage."""
from pathlib import Path
import time
import h5py
import numpy as np
from astropy.coordinates import SkyCoord
from sesnaimpute import config as config_module
from .constants import BAND_KEYS, N_POS
from .geometry import (MapGrid, DetectedPacsNeighbours, admitted_sample_points,
                       regional_cell_ids, certify_cell_empty)
from .stage import usable_xy
from .regional import regional_source_arrays, iter_owned_group_targets, select_sample_maps, select_sources
from .regional_workers import iterate_cell_calibrations, iterate_cell_recoveries
from .completeness import canonical_grid, CurveTable, pixel_curves
from .photometry import apply_group_target
from .products import _geometry_arrays, _put_group_geometry, write_products


def finish_regional_products(config, region, regional_by_band, *, staged_bands=None,
                              inventory_complete, absent_bands=(), publish=False,
                              calibration_seed=87421, curve_seed=471027,
                              area_draws=1024, max_draws=1048576, n_jobs=1):
    """Finish converged bands and merge with already validated band stages.

Existing P70 work may be passed in ``staged_bands`` while new P100 maps use
``regional_by_band``. A band cannot be supplied twice. The merge checks map
inventory completeness before canonical publication; staging never invents an
unmeasured band's coverage.
    """
    from .products import merge_staged_bands
    stages=dict(staged_bands or {})
    if set(stages)&set(regional_by_band):
        raise ValueError('A band cannot have both a retained stage and new regional calculation')
    supplied=set(stages)|set(regional_by_band)
    if not supplied or not supplied<=set(BAND_KEYS):
        raise ValueError('Supply actual PACS bands')
    if set(absent_bands)&supplied or not set(absent_bands)<=set(BAND_KEYS):
        raise ValueError('Absent bands must be known and disjoint from measured bands')
    if publish and not inventory_complete:
        raise ValueError('Canonical publication requires complete observation inventory')
    for band in BAND_KEYS:
        if band not in regional_by_band:
            continue
        regional=regional_by_band[band]
        if any(m.opened.record.band!=band for m in regional.maps):
            raise ValueError('Regional stage keyed by the wrong band')
        stages[band]=finish_regional_band(config,region,regional,
            calibration_seed=calibration_seed,curve_seed=curve_seed,
            area_draws=area_draws,max_draws=max_draws,n_jobs=n_jobs)
    return merge_staged_bands(config,region,stages,inventory_complete=inventory_complete,
                              absent_bands=absent_bands,publish=publish)


def finish_regional_band(config, region, regional, *, calibration_seed, curve_seed,
                         area_draws, max_draws, n_jobs):
    """Write a validated band stage for subsequent ``merge_staged_bands``.

This never publishes canonical regional products. Inventory/both-band gates
remain in the merge writer. Every native row uses the regional frozen mask;
only selected-map targets contribute source rows. HEALPix samples select their
own minimum-total-sigma map and link exactly to its singleton curve.
"""
    if calibration_seed == curve_seed:
        raise ValueError('Calibration and recovery streams must be distinct')
    if any(s.calibration_seed != calibration_seed for s in regional.stages):
        raise ValueError('Calibration seed differs from regional stage')
    if len(regional.maps) != len(regional.stages) or not regional.maps:
        raise ValueError('Matching nonempty regional maps and stages required')
    map_ids=[m.opened.record.map_id for m in regional.maps]
    if map_ids!=sorted(set(map_ids)):
        raise ValueError('Regional maps must retain sorted unique ownership order')
    candidates=regional.stages[0].candidates
    for stage in regional.stages:
        for name in ('ra_deg','dec_deg','source_row','catalogue'):
            if not np.array_equal(getattr(stage.candidates,name),getattr(candidates,name)):
                raise ValueError('Regional maps have inconsistent candidate row identities')
    owner,flux,sigma,detected=select_sources(regional.stages)
    for actual,expected in ((regional.source_map,owner),(regional.flux_mjy,flux),
                            (regional.sigma_mjy,sigma),(regional.detected,detected)):
        if not np.array_equal(actual,expected,equal_nan=True):
            raise ValueError('Regional ownership/detections are not a calibrated fixed point')
    bands={m.opened.record.band for m in regional.maps}
    if len(bands)!=1:
        raise ValueError('Finish one ownership-resolved band at a time')
    band=bands.pop();column=BAND_KEYS.index(band)
    source_path=config_module.product_path(config,'catalog','sesna','sources','source',region)
    with h5py.File(source_path,'r') as handle:
        names=np.asarray(handle['NAME'])
    admitted_path=config_module.product_path(config,'catalog','sesna','completeness','source',region)
    with h5py.File(admitted_path,'r') as handle:
        admitted=np.asarray(handle['HPX_PIX_512'])
    photo,mappings=regional_source_arrays(regional,Path(source_path).name,names.size)
    photo['NAME']=names
    curves={'HPX_PIX_512':admitted,'SOURCE_ROW':np.full((names.size,len(BAND_KEYS)),-1,np.int32)}
    magnitude,fluxgrid=canonical_grid();table=CurveTable(fluxgrid)
    pra,pdec=admitted_sample_points(admitted)
    sky=SkyCoord(pra,pdec,unit='deg',frame='icrs')
    c=regional.stages[0].candidates;det=regional.detected
    neighbours=DetectedPacsNeighbours(c,det)
    geometries=[];lookups=[];calibration_cells=[];calibration_scatters=[]
    for map_index,(m,stage,mapping) in enumerate(zip(regional.maps,regional.stages,mappings)):
        if not np.array_equal(stage.detected,det):
            raise ValueError('Per-map stage does not retain the shared detected set')
        opened=m.opened;record=opened.record
        grid=MapGrid.from_wcs(opened.wcs,opened.image.shape)
        def usable(x,y):
            return usable_xy(opened,x,y,half_size=m.half_size,min_coverage=m.min_coverage)
        px,py=opened.wcs.world_to_pixel(sky)
        covered=usable(px,py);pixel_cells=grid.cell_ids(px,py)
        local=(mapping>=0)&stage.covered
        extra=np.r_[stage.cell_ids[local],pixel_cells[covered]]
        cells=regional_cell_ids(grid,admitted,np.unique(extra))
        geometry=_geometry_arrays(band,cells.size)
        geometry['CALIBRATION_SEED_'+band]=np.asarray(calibration_seed,np.uint64)
        geometry['CURVE_SEED_'+band]=np.asarray(curve_seed,np.uint64)
        geometry['CELL_ID_'+band]=cells
        geometry['MAP_ID_'+band][:]=record.map_id
        geometry['OBSID_'+band][:]=','.join(str(v) for v in record.obsids)
        geometry['RA_DEG_'+band],geometry['DEC_DEG_'+band]=grid.centres_icrs(cells)
        geometry['AREA_N_CANDIDATE_'+band][:]=area_draws
        x0,x1,y0,y1=grid.bounds(cells)
        eligible_centre=usable((x0+x1)/2,(y0+y1)/2)&~neighbours.excluded(
            geometry['RA_DEG_'+band],geometry['DEC_DEG_'+band],m.beam_arcsec)
        empty=np.zeros(cells.size,bool)
        for i in np.flatnonzero(~eligible_centre):
            empty[i]=certify_cell_empty(grid,int(cells[i]),neighbours,usable,m.beam_arcsec)
        geometry['EMPTY_CERTIFIED_'+band]=empty
        active=cells[~empty]
        print('PACS regional map',record.map_id,'native cells',cells.size,'active',active.size,flush=True)
        for i in np.flatnonzero(empty):
            x,y=grid.sample_pixels(cells[i:i+1],area_draws,curve_seed)
            cov=usable(x,y);points=grid.wcs.pixel_to_world(x,y).icrs
            exc=neighbours.excluded(points.ra.deg,points.dec.deg,m.beam_arcsec)
            nc=int(cov.sum());ne=int((cov&exc).sum())
            geometry['AREA_N_COVERED_'+band][i]=nc
            geometry['AREA_N_EXCLUDED_'+band][i]=ne
            geometry['AREA_COVERED_FRAC_'+band][i]=nc/area_draws
            geometry['AREA_EXCLUDED_FRAC_'+band][i]=ne/nc if nc else np.nan
        means=dict(zip(stage.calibration_cell_ids,stage.calibration_mean_mjy))
        scatters=dict(zip(stage.calibration_cell_ids,stage.calibration_scatter_mjy))
        missing=np.asarray([cell for cell in active if cell not in scatters],np.int64)
        key=('regional-products',record.map_id,time.monotonic_ns())
        shared=dict(n_jobs=n_jobs,run_key=key,pass_key=('frozen',key),area_draws=area_draws,
                    max_draws=max_draws,half_size=m.half_size,annulus_arcsec=m.annulus_arcsec,
                    min_coverage=m.min_coverage,beam_arcsec=m.beam_arcsec,n_positions=N_POS,batch_size=N_POS)
        completed=0;last=time.monotonic()
        for piece in iterate_cell_calibrations(config,record,m.psf,missing,c.ra_deg[det],c.dec_deg[det],
                                               seed=calibration_seed,**shared):
            means.update(zip(piece.cell_ids,piece.mean_mjy));scatters.update(zip(piece.cell_ids,piece.scatter_mjy))
            completed+=piece.cell_ids.size
            if time.monotonic()-last>=30 or completed==missing.size:
                print('PACS regional native calibration',record.map_id,completed,'of',missing.size,flush=True)
                last=time.monotonic()
        lookup=np.full(grid.n_cells,-1,np.int32)
        completed=0;last=time.monotonic()
        for piece in iterate_cell_recoveries(config,record,m.psf,active,c.ra_deg[det],c.dec_deg[det],
                                             seed=curve_seed,**shared):
            scatter=np.asarray([scatters[cell] for cell in piece.cell_ids])
            sigma=np.hypot(piece.pixel_sigma_mjy,scatter[:,None])
            rows=table.append(piece.flux_mjy,sigma,piece.gain,zero_flux_statistic=piece.flux_mjy/sigma)
            lookup[piece.cell_ids]=rows;i=np.searchsorted(cells,piece.cell_ids)
            geometry['CURVE_ROW_'+band][i]=rows
            geometry['LOCAL_RMS_'+band+'_MJY'][i]=np.sqrt(np.mean(sigma*sigma,axis=1))
            geometry['GAIN_'+band][i]=piece.gain.mean(axis=1)
            geometry['BACKGROUND_MEAN_'+band+'_MJY'][i]=[means[cell] for cell in piece.cell_ids]
            geometry['CELL_SCATTER_'+band+'_MJY'][i]=scatter
            geometry['AREA_N_COVERED_'+band][i]=piece.n_covered
            geometry['AREA_N_EXCLUDED_'+band][i]=piece.n_excluded
            geometry['AREA_COVERED_FRAC_'+band][i]=piece.n_covered/area_draws
            geometry['AREA_EXCLUDED_FRAC_'+band][i]=np.divide(piece.n_excluded,piece.n_covered,
                out=np.full(piece.cell_ids.size,np.nan),where=piece.n_covered>0)
            completed+=piece.cell_ids.size
            if time.monotonic()-last>=30 or completed==active.size:
                print('PACS regional native curves',record.map_id,completed,'of',active.size,flush=True)
                last=time.monotonic()
        curves['SOURCE_ROW'][mapping[local],column]=lookup[stage.cell_ids[local]]
        geometries.append(geometry);lookups.append(lookup)
        cc=np.asarray(sorted(scatters),np.int64)
        calibration_cells.append(cc);calibration_scatters.append(np.asarray([scatters[cell] for cell in cc]))
    # Map-local CELL_ID is intentionally repeated across maps. Curve row IDs
    # are global within this band and every audit retains its MAP_ID.
    geometry=_geometry_arrays(band,0)
    for key in geometries[0]:
        if key.startswith('GROUP_'):
            continue
        geometry[key]=(geometries[0][key] if np.asarray(geometries[0][key]).ndim==0
                       else np.concatenate([g[key] for g in geometries]))
    groups=[]
    for map_index,result in iter_owned_group_targets(regional,calibration_seed=calibration_seed,
            curve_seed=curve_seed,area_draws=area_draws,max_draws=max_draws,n_positions=N_POS):
        mapping=mappings[map_index];record=regional.maps[map_index].opened.record
        target_local,inseparable=apply_group_target(photo,mapping,result,band)
        target=result.target_source
        if len(inseparable):
            curves['SOURCE_ROW'][inseparable,column]=-1
            target_local=int(mapping[result.members[target]])
            row=-1;s=result.curve_measurement.group_sigma[None,:]
            real_sigma=result.source_measurement.group_sigma[0]
            background_mean=result.calibration_mean_mjy.sum();gain=np.nan
        else:
            if target_local<0:
                raise RuntimeError('Non-owned target reached regional group output')
            f=result.curve_measurement.flux[:,target][None,:]
            s=result.curve_measurement.sigma[:,target][None,:]
            row=int(table.append(f,s,result.gain[None,:],zero_flux_statistic=f/s)[0])
            curves['SOURCE_ROW'][target_local,column]=row
            real_sigma=result.source_measurement.sigma[0,target]
            background_mean=result.calibration_mean_mjy[target];gain=float(result.gain.mean())
        groups.append(dict(GROUP_CURVE_ROW=row,GROUP_MAP_ID=record.map_id,GROUP_ID=result.group_id,
            GROUP_TARGET_SOURCE_ROW=target_local,GROUP_ANCHOR_CELL_ID=result.target_cell_id,
            GROUP_N_MEMBER=len(result.members),GROUP_REAL_SIGMA=real_sigma,
            GROUP_RANDOM_SIGMA_MIN=float(s.min()),GROUP_RANDOM_SIGMA_MAX=float(s.max()),
            GROUP_GAIN=gain,GROUP_BACKGROUND_MEAN=background_mean,
            GROUP_AREA_N_CANDIDATE=int(result.area.n_candidate[0]),
            GROUP_AREA_N_COVERED=int(result.area.n_covered[0]),GROUP_AREA_N_EXCLUDED=int(result.area.n_excluded[0]),
            members_source_row=c.source_row[result.members],members_catalogue=c.catalogue[result.members],
            covariance=result.calibration_covariance_mjy2))
    _put_group_geometry(geometry,band,groups)
    arrays=table.arrays()
    owner,_,_,excluded,prows=select_sample_maps(regional,pra.ravel(),pdec.ravel(),calibration_cells,
                                               calibration_scatters,curve_lookups=lookups)
    pixels=pixel_curves(arrays['curves'],prows.reshape(pra.shape),(owner>=0).reshape(pra.shape),excluded.reshape(pra.shape))
    curves.update({'CURVE_'+band:arrays['curves'],'MAG_GRID_'+band:magnitude,
        'N_POS_'+band:arrays['n_positions'],'COMPLETED_FROM_'+band:arrays['completed_from'],
        'ENVELOPE_'+band:arrays['envelope'],'PIXEL_CURVE_'+band:pixels.curves,
        'PIXEL_COVERED_FRAC_'+band:pixels.covered_fraction,'PIXEL_EXCLUDED_FRAC_'+band:pixels.excluded_fraction})
    return write_products(config,region,photo,curves,geometry,publish=False,complete=False)
