"""Standalone PACS science products and schema/readback validation."""
from pathlib import Path
import os
import tempfile

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute.attrs_registry import REGISTRY
from .constants import (BAND_KEYS, N_POS, STORAGE_SCALE, ZERO_POINT_JY, UPPER_LIMIT_SIGMA,
                        P70_PARALLEL60_GROUP_RADIUS_ARCSEC)
from .maps import checked_path


def _require(data, key, shape, dtype=None):
    if key not in data:
        raise ValueError('Missing PACS dataset '+key)
    value=np.asarray(data[key])
    if value.shape!=shape or (dtype is not None and value.dtype!=np.dtype(dtype)):
        raise ValueError('%s requires shape %s dtype %s, got %s/%s'%(key,shape,dtype,value.shape,value.dtype))
    return value


def validate_products(photometry, curves, geometry,*,bands=BAND_KEYS):
    """Cross-check source identities, conditional curves, and native row links."""
    names=np.asarray(photometry['NAME'])
    if names.ndim!=1:
        raise ValueError('NAME must be the original one-dimensional source order')
    nsource=names.size
    admitted=np.asarray(curves['HPX_PIX_512'])
    if admitted.ndim!=1 or not np.issubdtype(admitted.dtype,np.integer) or np.unique(admitted).size!=admitted.size:
        raise ValueError('HPX_PIX_512 must preserve unique admitted pixel IDs')
    source_rows=_require(curves,'SOURCE_ROW',(nsource,len(BAND_KEYS)),np.int32)
    from .completeness import canonical_grid
    canonical,_=canonical_grid()
    for band in bands:
        column=BAND_KEYS.index(band)
        mag=np.asarray(curves['MAG_GRID_'+band])
        if mag.ndim!=1 or mag.size<2 or not np.all(np.isfinite(mag)) or not np.all(np.diff(mag)>0):
            raise ValueError('Invalid magnitude grid '+band)
        if not np.allclose(np.diff(mag),mag[1]-mag[0],rtol=0,atol=1e-12):
            raise ValueError('PACS magnitude grid must be uniform')
        if mag.dtype!=np.float64 or not np.array_equal(mag,canonical):
            raise ValueError('Published magnitude grid must equal the fixed band constants')
        table=np.asarray(curves['CURVE_'+band])
        if table.ndim!=2 or table.shape[1]!=mag.size or table.dtype!=np.uint16:
            raise ValueError('Invalid encoded curve table '+band)
        nrow=table.shape[0]
        if np.any(table>STORAGE_SCALE) or np.any(np.diff(table.astype(np.int32),axis=1)>0):
            raise ValueError('Raw curves must be bounded and nonincreasing with magnitude')
        if np.any(table[:,0]!=STORAGE_SCALE) or np.any(table%(STORAGE_SCALE//N_POS)):
            raise ValueError('Curve plateau/quantization disagrees with full accepted-count sampling')
        count=_require(curves,'N_POS_'+band,(nrow,),np.int16)
        if np.any(count!=N_POS):
            raise ValueError('Every published curve requires the full accepted sample')
        completed=_require(curves,'COMPLETED_FROM_'+band,(nrow,),np.float32)
        if not np.all(np.isnan(completed)):
            raise ValueError('PACS curves must not contain extrapolated segments')
        envelope=_require(curves,'ENVELOPE_'+band,(mag.size,),np.uint16)
        expected=table.max(axis=0) if nrow else np.zeros(mag.size,np.uint16)
        if not np.array_equal(envelope,expected):
            raise ValueError('Envelope does not match measured rows')
        pixel=_require(curves,'PIXEL_CURVE_'+band,(admitted.size,mag.size),np.float32)
        covered=_require(curves,'PIXEL_COVERED_FRAC_'+band,(admitted.size,),np.float32)
        excluded=_require(curves,'PIXEL_EXCLUDED_FRAC_'+band,(admitted.size,),np.float32)
        if np.any(~np.isfinite(covered)|(covered<0)|(covered>1)):
            raise ValueError('Invalid pixel coverage fractions')
        if np.any((covered>0)&(~np.isfinite(excluded)|(excluded<0)|(excluded>1))):
            raise ValueError('Covered pixels require explicit exclusion fractions')
        if np.any((covered==0)&~np.isnan(excluded)) or np.any(pixel[covered==0]!=0):
            raise ValueError('Uncovered pixel convention violated')
        empty=(covered>0)&(excluded==1)
        if not np.all(np.isnan(pixel[empty])):
            raise ValueError('Fully excluded covered pixels must have no conditional curve')
        present=(covered>0)&(excluded<1)
        if np.any(~np.isfinite(pixel[present])|(pixel[present]<0)|(pixel[present]>1)):
            raise ValueError('Accepted pixel samples require real finite curves')
        origin=_require(photometry,'ORIGIN_'+band,(nsource,),np.int8)
        _require(photometry,'FIRST_PASS_DETECTED_'+band,(nsource,),np.bool_)
        if not np.all(np.isin(origin,[1,2,90,91])):
            raise ValueError('Unknown PACS origin')
        f=_require(photometry,'FNU_'+band+'_MJY',(nsource,),np.float64)
        s=_require(photometry,'SIGMA_'+band+'_MJY',(nsource,),np.float64)
        for prefix,dtype in [('LOCAL_RMS_',np.float32),('N_BLEND_',np.int16),
                             ('BLEND_GROUP_',np.int32),('EXTENDED_',np.bool_)]:
            key=prefix+band+('_MJY' if prefix=='LOCAL_RMS_' else '')
            _require(photometry,key,(nsource,),dtype)
        _require(photometry,'OBSID_'+band,(nsource,))
        if np.any((source_rows[:,column]<-1)|(source_rows[:,column]>=nrow)):
            raise ValueError('Source completeness row outside its band table')
        if np.any(np.isin(origin,[2,91])&(source_rows[:,column]!=-1)):
            raise ValueError('Excluded bounds/off-map sources must not acquire likelihood curves')
        if np.any((origin==90)&(source_rows[:,column]<0)):
            raise ValueError('Covered non-detection lacks a measured matching curve')
        measured=np.isin(origin,[1,90])
        if np.any(measured&(~np.isfinite(f)|~np.isfinite(s)|(s<=0))):
            raise ValueError('Measured sources require finite flux and positive uncertainty')
        if np.any((origin==1)&(f/s<5)) or np.any((origin==90)&(f/s>=5)):
            raise ValueError('Origin disagrees with the calibrated detection statistic')
        bounds=origin==2
        if np.any(bounds&(~np.isfinite(f)|(f<0)|(s!=UPPER_LIMIT_SIGMA))):
            raise ValueError('Stored inseparable bounds require finite nonnegative flux and bound sentinel')
        cells=np.asarray(geometry['CELL_ID_'+band])
        if cells.ndim!=1 or not np.issubdtype(cells.dtype,np.integer):
            raise ValueError('Native geometry cell IDs must be an integer vector')
        maps=_require(geometry,'MAP_ID_'+band,cells.shape).astype('U')
        if len(set(zip(maps.tolist(),cells.tolist())))!=cells.size:
            raise ValueError('Native geometry (MAP_ID,CELL_ID) pairs must be unique')
        links=_require(geometry,'CURVE_ROW_'+band,cells.shape,np.int32)
        certified=_require(geometry,'EMPTY_CERTIFIED_'+band,cells.shape,np.bool_)
        calibration_seed=_require(geometry,'CALIBRATION_SEED_'+band,(),np.uint64)
        curve_seed=_require(geometry,'CURVE_SEED_'+band,(),np.uint64)
        if cells.size and calibration_seed==curve_seed:
            raise ValueError('Calibration and held-out recovery streams must be distinct')
        if np.any((links<-1)|(links>=nrow)):
            raise ValueError('Native cell row link outside curve table')
        if np.any(certified&(links!=-1)) or np.any((links==-1)&~certified):
            raise ValueError('A missing native curve requires a geometric empty-area certificate')
        if np.unique(links[links>=0]).size!=np.count_nonzero(links>=0):
            raise ValueError('Distinct native cells must not share a curve row')
        for name in ('RA_DEG','DEC_DEG','GAIN','AREA_COVERED_FRAC','AREA_EXCLUDED_FRAC'):
            _require(geometry,name+'_'+band,cells.shape,np.float64)
        for name in ('LOCAL_RMS','BACKGROUND_MEAN','CELL_SCATTER'):
            _require(geometry,name+'_'+band+'_MJY',cells.shape,np.float64)
        _require(geometry,'OBSID_'+band,cells.shape)
        candidates=_require(geometry,'AREA_N_CANDIDATE_'+band,cells.shape,np.int64)
        covered_count=_require(geometry,'AREA_N_COVERED_'+band,cells.shape,np.int64)
        excluded_count=_require(geometry,'AREA_N_EXCLUDED_'+band,cells.shape,np.int64)
        if np.any((candidates<=0)|(covered_count<0)|(covered_count>candidates)|
                  (excluded_count<0)|(excluded_count>covered_count)):
            raise ValueError('Native area counts are inconsistent')
        if np.any(certified&(covered_count!=excluded_count)):
            raise ValueError('Certified-empty cell has an accepted fixed-area sample')
        expected_covered=np.divide(covered_count,candidates,out=np.zeros(cells.size),where=candidates>0)
        expected_excluded=np.divide(excluded_count,covered_count,out=np.full(cells.size,np.nan),where=covered_count>0)
        np.testing.assert_allclose(geometry['AREA_COVERED_FRAC_'+band],expected_covered,rtol=0,atol=1e-12)
        np.testing.assert_allclose(geometry['AREA_EXCLUDED_FRAC_'+band],expected_excluded,rtol=0,atol=1e-12,equal_nan=True)
        measured_cell=links>=0
        for key in ('LOCAL_RMS_'+band+'_MJY','GAIN_'+band):
            values=geometry[key][measured_cell]
            if np.any(~np.isfinite(values)|(values<=0)):
                raise ValueError('Measured native curve requires positive finite RMS and gain')
        group_rows=np.asarray(geometry['GROUP_CURVE_ROW_'+band])
        if group_rows.ndim!=1 or group_rows.dtype!=np.int32 or np.any((group_rows<-1)|(group_rows>=nrow)):
            raise ValueError('Invalid group curve-row audit')
        ngroup=group_rows.size
        _require(geometry,'GROUP_MAP_ID_'+band,(ngroup,))
        members=_require(geometry,'GROUP_N_MEMBER_'+band,(ngroup,),np.int16)
        if np.any(members<2):
            raise ValueError('Joint-group audit requires at least two physical members')
        for name,dtype in [('GROUP_ID',np.int32),('GROUP_TARGET_SOURCE_ROW',np.int64),
                           ('GROUP_ANCHOR_CELL_ID',np.int64),('GROUP_GAIN',np.float64),
                           ('GROUP_AREA_N_CANDIDATE',np.int64),('GROUP_AREA_N_COVERED',np.int64),
                           ('GROUP_AREA_N_EXCLUDED',np.int64)]:
            _require(geometry,name+'_'+band,(ngroup,),dtype)
        for name in ('GROUP_REAL_SIGMA','GROUP_RANDOM_SIGMA_MIN','GROUP_RANDOM_SIGMA_MAX','GROUP_BACKGROUND_MEAN'):
            _require(geometry,name+'_'+band+'_MJY',(ngroup,),np.float64)
        mo=_require(geometry,'GROUP_MEMBER_OFFSET_'+band,(ngroup+1,),np.int64)
        co=_require(geometry,'GROUP_COVARIANCE_OFFSET_'+band,(ngroup+1,),np.int64)
        if mo[0]!=0 or co[0]!=0 or not np.array_equal(np.diff(mo),members) or not np.array_equal(np.diff(co),members.astype(np.int64)**2):
            raise ValueError('Ragged group members/covariance offsets disagree')
        _require(geometry,'GROUP_MEMBER_SOURCE_ROW_'+band,(int(mo[-1]),),np.int64)
        _require(geometry,'GROUP_MEMBER_CATALOGUE_'+band,(int(mo[-1]),))
        _require(geometry,'GROUP_CELL_COVARIANCE_'+band+'_MJY2',(int(co[-1]),),np.float64)
        all_rows=np.r_[links[links>=0],group_rows[group_rows>=0]]
        if not np.array_equal(np.sort(all_rows),np.arange(nrow)):
            raise ValueError('Every curve row requires exactly one native or group-target audit link')
    for stem,data in [('photometry_pacs_source',photometry),('completeness_pacs_source',curves),
                      ('geometry_pacs_region',geometry)]:
        for key in data:
            if (stem,key) not in REGISTRY:
                raise KeyError('Unregistered PACS dataset: '+stem+'/'+key)




def readback_products(paths, *, bands=BAND_KEYS, expected=None):
    """Validate standalone HDF5 arrays, attributes and exact stored values.

    No downstream consumer is imported or invoked. Comparing the complete
    stored curve table to the measured input preserves every empirical floor
    and endpoint; this check neither caps nor extrapolates probabilities.
    """
    stems=('photometry_pacs_source','completeness_pacs_source','geometry_pacs_region')
    if len(paths)!=3 or (expected is not None and len(expected)!=3):
        raise ValueError('Three standalone PACS products are required')
    loaded=[]
    for index,(path,stem) in enumerate(zip(paths,stems)):
        with h5py.File(path,'r') as handle:
            data={}
            if expected is not None and set(handle)!=set(expected[index]):
                raise ValueError('Stored PACS dataset names differ from measured arrays')
            for key,ds in handle.items():
                units,reading=REGISTRY[(stem,key)]
                if ds.attrs.get('UNITS')!=units or ds.attrs.get('READING')!=reading:
                    raise ValueError('Stored PACS attributes differ from registry: '+key)
                if key.startswith('CURVE_') and ds.attrs.get('ZERO_POINT_JY')!=ZERO_POINT_JY:
                    raise ValueError('Stored PACS curve zero point differs')
                value=np.asarray(ds)
                data[key]=value
                if expected is not None:
                    original=np.asarray(expected[index][key])
                    if value.dtype.kind in 'OSU' or original.dtype.kind in 'OSU':
                        equal=np.array_equal(value.astype('U'),original.astype('U'))
                    else:
                        equal=(value.dtype==original.dtype and
                               np.array_equal(value,original,equal_nan=True))
                    if not equal:
                        raise ValueError('Stored PACS values differ from measured arrays: '+key)
            loaded.append(data)
    validate_products(*loaded,bands=bands)
    count=sum(loaded[1]['CURVE_'+band].shape[0] for band in bands)
    print('PACS standalone HDF5 readback passed',count,'measured curve rows; raw floors unchanged',flush=True)
    return count


def write_products(config,region,photometry,curves,geometry,*,publish=False,complete=True):
    """Write three fully validated final products beneath the checked PACS root."""
    if publish and not complete:
        raise ValueError('An incomplete current-map product cannot be published')
    bands=BAND_KEYS if complete else tuple(b for b in BAND_KEYS if 'CURVE_'+b in curves)
    if not bands:
        raise ValueError('No measured band to write')
    validate_products(photometry,curves,geometry,bands=bands)
    source_path=config_module.product_path(config,'catalog','sesna','sources','source',region)
    with h5py.File(source_path,'r') as source:
        actual=np.asarray(source['NAME']).astype('U')
        if not np.array_equal(np.asarray(photometry['NAME']).astype('U'),actual):
            raise ValueError('Photometry NAME order differs from the actual SESNA catalogue')
    admitted_path=config_module.product_path(config,'catalog','sesna','completeness','source',region)
    with h5py.File(admitted_path,'r') as source:
        if not np.array_equal(curves['HPX_PIX_512'],source['HPX_PIX_512'][:]):
            raise ValueError('Admitted pixel order differs from the actual SESNA completeness file')
    pending=[]
    try:
        for quantity,granule,data in [('photometry','source',photometry),
                                      ('completeness','source',curves),('geometry','region',geometry)]:
            stem=quantity+'_pacs_'+granule
            requested=Path(config_module.product_path(config,'sky/derived','pacs',quantity,granule,region))
            path=checked_path(Path(config.data_root),str(requested.relative_to(config.data_root)))
            expected=checked_path(Path(config.data_root),'sky/derived/pacs')
            if path.parent!=expected:
                raise ValueError('Unexpected PACS product destination')
            path.parent.mkdir(parents=True,exist_ok=True)
            descriptor,temp=tempfile.mkstemp(prefix='.'+path.name+'.'+'-'.join(bands)+'.',suffix='.part',dir=path.parent)
            os.close(descriptor);pending.append((Path(temp),path))
            with h5py.File(temp,'w') as out:
                for key,value in data.items():
                    array=np.asarray(value)
                    if array.dtype.kind in 'OU':
                        array=np.asarray(array,dtype=h5py.string_dtype('utf-8'))
                    ds=out.create_dataset(key,data=array)
                    ds.attrs['UNITS'],ds.attrs['READING']=REGISTRY[(stem,key)]
                    if key.startswith('CURVE_'):
                        ds.attrs['ZERO_POINT_JY']=ZERO_POINT_JY
        readback_products(tuple(p for p,d in pending), bands=bands, expected=(photometry,curves,geometry))
        if not publish:
            staged=tuple(temporary for temporary,destination in pending)
            for temporary,destination in pending:
                print('PACS validated staged '+('regional' if complete else 'CURRENT-MAP PARTIAL')+
                      ' product',temporary,'intended for',destination,flush=True)
            pending=[]
            return staged
        for temporary,destination in pending:
            # Recheck destination immediately before publishing.
            checked_path(Path(config.data_root),str(destination.relative_to(config.data_root)))
            os.replace(temporary,destination)
            print('PACS wrote',destination,flush=True)
        return tuple(destination for temporary,destination in pending)
    finally:
        for temporary,destination in pending:
            if temporary.exists():
                temporary.unlink()


def _empty_band(band,nsource,npixel,magnitude):
    """Only call after observation inventory establishes this band is absent."""
    photo={
        'FNU_'+band+'_MJY':np.full(nsource,np.nan,np.float64),
        'SIGMA_'+band+'_MJY':np.full(nsource,np.nan,np.float64),
        'ORIGIN_'+band:np.full(nsource,91,np.int8),
        'LOCAL_RMS_'+band+'_MJY':np.full(nsource,np.nan,np.float32),
        'OBSID_'+band:np.full(nsource,b'',dtype='S1'),
        'N_BLEND_'+band:np.zeros(nsource,np.int16),
        'BLEND_GROUP_'+band:np.full(nsource,-1,np.int32),
        'EXTENDED_'+band:np.zeros(nsource,bool),
        'FIRST_PASS_DETECTED_'+band:np.zeros(nsource,bool)}
    curves={'CURVE_'+band:np.empty((0,magnitude.size),np.uint16),
        'MAG_GRID_'+band:magnitude.copy(),'N_POS_'+band:np.empty(0,np.int16),
        'COMPLETED_FROM_'+band:np.empty(0,np.float32),
        'ENVELOPE_'+band:np.zeros(magnitude.size,np.uint16),
        'PIXEL_CURVE_'+band:np.zeros((npixel,magnitude.size),np.float32),
        'PIXEL_COVERED_FRAC_'+band:np.zeros(npixel,np.float32),
        'PIXEL_EXCLUDED_FRAC_'+band:np.full(npixel,np.nan,np.float32)}
    return photo,curves,_geometry_arrays(band,0)


def _geometry_arrays(band,ncell):
    result={}
    for prefix in ('CELL_ID','AREA_N_CANDIDATE','AREA_N_COVERED','AREA_N_EXCLUDED'):
        result[prefix+'_'+band]=np.zeros(ncell,np.int64)
    result['CURVE_ROW_'+band]=np.full(ncell,-1,np.int32)
    result['EMPTY_CERTIFIED_'+band]=np.zeros(ncell,bool)
    result['CALIBRATION_SEED_'+band]=np.asarray(0,np.uint64)
    result['CURVE_SEED_'+band]=np.asarray(0,np.uint64)
    for prefix in ('RA_DEG','DEC_DEG','GAIN','AREA_COVERED_FRAC','AREA_EXCLUDED_FRAC'):
        result[prefix+'_'+band]=np.full(ncell,np.nan,np.float64)
    for prefix in ('LOCAL_RMS','BACKGROUND_MEAN','CELL_SCATTER'):
        result[prefix+'_'+band+'_MJY']=np.full(ncell,np.nan,np.float64)
    for prefix in ('MAP_ID','OBSID'):
        result[prefix+'_'+band]=np.full(ncell,'',dtype=object)
    _put_group_geometry(result,band,[])
    return result


def _put_group_geometry(geometry,band,entries):
    """Flatten target-specific joint calibration audits without padded matrices."""
    scalar={
        'GROUP_CURVE_ROW':np.int32,'GROUP_ID':np.int32,
        'GROUP_TARGET_SOURCE_ROW':np.int64,'GROUP_ANCHOR_CELL_ID':np.int64,
        'GROUP_N_MEMBER':np.int16,'GROUP_GAIN':np.float64,
        'GROUP_AREA_N_CANDIDATE':np.int64,'GROUP_AREA_N_COVERED':np.int64,
        'GROUP_AREA_N_EXCLUDED':np.int64}
    for name,dtype in scalar.items():
        geometry[name+'_'+band]=np.array([entry[name] for entry in entries],dtype=dtype)
    geometry['GROUP_MAP_ID_'+band]=np.asarray([entry['GROUP_MAP_ID'] for entry in entries],dtype=object)
    for name in ('GROUP_REAL_SIGMA','GROUP_RANDOM_SIGMA_MIN','GROUP_RANDOM_SIGMA_MAX','GROUP_BACKGROUND_MEAN'):
        geometry[name+'_'+band+'_MJY']=np.array([entry[name] for entry in entries],np.float64)
    member_offsets=[0];covariance_offsets=[0];rows=[];catalogues=[];covariance=[]
    for entry in entries:
        rows.extend(entry['members_source_row']);catalogues.extend(entry['members_catalogue'])
        covariance.extend(np.asarray(entry['covariance']).ravel())
        member_offsets.append(len(rows));covariance_offsets.append(len(covariance))
    geometry['GROUP_MEMBER_OFFSET_'+band]=np.asarray(member_offsets,np.int64)
    geometry['GROUP_MEMBER_SOURCE_ROW_'+band]=np.asarray(rows,np.int64)
    geometry['GROUP_MEMBER_CATALOGUE_'+band]=np.asarray(catalogues,dtype=object)
    geometry['GROUP_COVARIANCE_OFFSET_'+band]=np.asarray(covariance_offsets,np.int64)
    geometry['GROUP_CELL_COVARIANCE_'+band+'_MJY2']=np.asarray(covariance,np.float64)


def finish_single_map(config,region,record,opened,cached_psf,exact_psf,singletons,*,
                      absent_bands,inventory_complete,n_jobs,half_size=12,
                      annulus_arcsec=(25.,35.),min_coverage=np.nextafter(0.,1.),
                      calibration_seed=87421,curve_seed=471027,area_draws=1024,
                      max_draws=1048576,publish=False,beam_arcsec=None,resume_state=None):
    """Finish Musca's measured map in the same process as calibrated first pass.

    Publication requires a completed archive inventory and explicit evidence for
    absent bands. This implementation rejects additional selected maps rather
    than silently inventing ownership or declaring an unprocessed band absent.
    """
    from .completeness import canonical_grid,CurveTable,pixel_curves
    from .geometry import (MapGrid,DetectedPacsNeighbours,regional_cell_ids,
                            admitted_sample_points,certify_cell_empty)
    from .stage import usable_xy,iter_group_targets
    from .photometry import source_product_arrays,apply_group_target
    from .regional_workers import iterate_cell_calibrations,iterate_cell_recoveries
    from .maps import read_index
    import time
    finish_started=time.monotonic()

    if publish and not inventory_complete:
        raise RuntimeError('Archive inventory incomplete: cannot publish regional coverage/products')
    records=read_index(config,region)
    same_band=[r for r in records if r.band==record.band]
    if set(absent_bands)&{r.band for r in records}:
        raise ValueError('Cannot declare an indexed observed band absent')
    if inventory_complete and (len(same_band)!=1 or same_band[0].map_id!=record.map_id):
        raise RuntimeError('Single-map band finalizer cannot omit or assign ownership for additional maps in its band')
    regional_complete=inventory_complete and set(absent_bands)==set(BAND_KEYS)-{record.band}
    if publish and not regional_complete:
        raise ValueError('Other observed bands must be assembled before regional publication')
    if beam_arcsec is None:
        if record.band!='P70' or record.mode!=31 or record.scan_speed!=60:
            raise ValueError('Supply an evidence-backed beam for this other band/mode')
        beam_arcsec=P70_PARALLEL60_GROUP_RADIUS_ARCSEC
    if calibration_seed==curve_seed:
        raise ValueError('Calibration and held-out curve streams must differ')
    if getattr(singletons,'calibration_seed',None)!=calibration_seed:
        raise ValueError('Supplied calibration seed differs from the retained singleton calibration')
    if resume_state is not None:
        for name,value in (('opened',opened),('cached_psf',cached_psf),('exact_psf',exact_psf),
                           ('singletons',singletons)):
            if resume_state.get(name) is not value:
                raise ValueError('Resume requires the identical retained '+name)
        for name,value in (('region',region),('record',record),('half_size',half_size),
                           ('annulus_arcsec',annulus_arcsec),('min_coverage',min_coverage),
                           ('calibration_seed',calibration_seed),('curve_seed',curve_seed),
                           ('area_draws',area_draws),('max_draws',max_draws)):
            if name not in resume_state or resume_state[name]!=value:
                raise ValueError('Resume parameter differs: '+name)
        if Path(resume_state['config'].data_root).resolve()!=Path(config.data_root).resolve():
            raise ValueError('Resume data root differs')
        if float(resume_state.get('beam',resume_state.get('beam_arcsec',np.nan)))!=float(beam_arcsec):
            raise ValueError('Resume exclusion beam differs')
        if 'det' in resume_state and not np.array_equal(resume_state['det'],singletons.detected):
            raise ValueError('Resume frozen detection mask differs')
    band=record.band;column=BAND_KEYS.index(band)
    source_path=config_module.product_path(config,'catalog','sesna','sources','source',region)
    with h5py.File(source_path,'r') as source:
        names=np.asarray(source['NAME'])
    admitted_path=config_module.product_path(config,'catalog','sesna','completeness','source',region)
    with h5py.File(admitted_path,'r') as source:
        admitted=np.asarray(source['HPX_PIX_512'])
    if resume_state is not None:
        if not np.array_equal(names,resume_state.get('names')) or not np.array_equal(admitted,resume_state.get('admitted')):
            raise ValueError('Resume catalogue/admitted pixel order changed')
    photo,mapping=source_product_arrays(singletons,Path(source_path).name,names.size,record)
    photo['NAME']=names
    curves={'HPX_PIX_512':admitted,'SOURCE_ROW':np.full((names.size,len(BAND_KEYS)),-1,np.int32)}
    magnitude,fluxgrid=canonical_grid()
    grid=MapGrid.from_wcs(opened.wcs,opened.image.shape)
    neighbours=DetectedPacsNeighbours(singletons.candidates,singletons.detected)
    beam=float(beam_arcsec)
    if not np.isfinite(beam) or beam<=0:
        raise ValueError('Positive finite evidence-backed beam required')
    def usable(x,y):
        return usable_xy(opened,x,y,half_size=half_size,min_coverage=min_coverage)
    pra,pdec=admitted_sample_points(admitted)
    from astropy.coordinates import SkyCoord
    psky=SkyCoord(pra,pdec,unit='deg',frame='icrs')
    px,py=opened.wcs.world_to_pixel(psky)
    pcovered=usable(px,py)
    pcells=grid.cell_ids(px,py)
    local=(mapping>=0)&singletons.covered
    extra=np.r_[singletons.cell_ids[local],pcells[pcovered]]
    cells=(np.asarray(resume_state['cells']) if resume_state is not None and 'cells' in resume_state
           else regional_cell_ids(grid,admitted,np.unique(extra)))
    print('PACS selected regional native cells',cells.size,flush=True)
    geometry=(resume_state['geometry'] if resume_state is not None and 'geometry' in resume_state
              else _geometry_arrays(band,cells.size))
    geometry['CALIBRATION_SEED_'+band]=np.asarray(calibration_seed,np.uint64)
    geometry['CURVE_SEED_'+band]=np.asarray(curve_seed,np.uint64)
    geometry['CELL_ID_'+band]=cells
    geometry['RA_DEG_'+band],geometry['DEC_DEG_'+band]=grid.centres_icrs(cells)
    geometry['MAP_ID_'+band][:]=record.map_id
    geometry['OBSID_'+band][:]=','.join(str(v) for v in record.obsids)
    geometry['AREA_N_CANDIDATE_'+band][:]=area_draws
    # A single usable, unexcluded centre disproves emptiness. Batch this common
    # case rather than performing tens of thousands of scalar sky transforms.
    if resume_state is not None and 'active' in resume_state:
        empty=np.asarray(resume_state['empty'],bool)
        if empty.shape!=cells.shape or not np.array_equal(cells[~empty],resume_state['active']):
            raise ValueError('Resume native cells/empty certificate mismatch')
    else:
        x0,x1,y0,y1=grid.bounds(cells)
        centre_x,centre_y=(x0+x1)/2,(y0+y1)/2
        centre_usable=usable(centre_x,centre_y)
        centre_excluded=neighbours.excluded(geometry['RA_DEG_'+band],geometry['DEC_DEG_'+band],beam)
        needs_proof=np.flatnonzero(~(centre_usable&~centre_excluded))
        empty=np.zeros(cells.size,bool)
        for done,i in enumerate(needs_proof,1):
            empty[i]=certify_cell_empty(grid,int(cells[i]),neighbours,usable,beam)
            if done%1000==0 or done==needs_proof.size:
                print('PACS native empty-area checks',done,'of',needs_proof.size,flush=True)
    geometry['EMPTY_CERTIFIED_'+band]=empty
    active=cells[~empty]
    # Even proved-empty cells retain their independent finite area estimate.
    for cell in cells[empty]:
        i=np.searchsorted(cells,cell);x,y=grid.sample_pixels(np.array([cell]),area_draws,curve_seed)
        covered=usable(x,y);sky=grid.wcs.pixel_to_world(x,y).icrs
        excluded=neighbours.excluded(sky.ra.deg,sky.dec.deg,beam)
        nc=int(covered.sum());ne=int((covered&excluded).sum())
        geometry['AREA_N_COVERED_'+band][i]=nc;geometry['AREA_N_EXCLUDED_'+band][i]=ne
        geometry['AREA_COVERED_FRAC_'+band][i]=nc/area_draws
        geometry['AREA_EXCLUDED_FRAC_'+band][i]=ne/nc if nc else np.nan
    print('PACS regional native cells',cells.size,'eligible',active.size,'certified empty',empty.sum(),flush=True)
    means=dict(zip(singletons.calibration_cell_ids,singletons.calibration_mean_mjy))
    scatters=dict(zip(singletons.calibration_cell_ids,singletons.calibration_scatter_mjy))
    if resume_state is not None:
        old_means=resume_state.get('means',{});old_scatters=resume_state.get('scatters',{})
        for cell in old_means.keys()&old_scatters.keys():
            mean,scatter=float(old_means[cell]),float(old_scatters[cell])
            if not np.isfinite(mean) or not np.isfinite(scatter) or scatter<0:
                raise ValueError('Resume calibration contains invalid measured statistics')
            if cell in means and (mean!=means[cell] or scatter!=scatters[cell]):
                raise ValueError('Resume changed retained first-pass calibration')
            means[cell]=mean;scatters[cell]=scatter
        print('PACS resume retained native calibrations',len(scatters),'elapsed',time.monotonic()-finish_started,flush=True)
    missing=np.array([cell for cell in active if cell not in scatters],np.int64)
    c=singletons.candidates;det=singletons.detected
    key=('final-map',record.map_id,time.monotonic_ns())
    shared=dict(n_jobs=n_jobs,run_key=key,pass_key=('frozen',key),area_draws=area_draws,
        max_draws=max_draws,half_size=half_size,annulus_arcsec=annulus_arcsec,
        min_coverage=min_coverage,beam_arcsec=beam,n_positions=N_POS,batch_size=N_POS)
    print('PACS missing native calibrations',missing.size,flush=True)
    calibrated=0
    last_progress=time.monotonic()
    for piece in iterate_cell_calibrations(config,record,cached_psf,missing,c.ra_deg[det],c.dec_deg[det],
                                           seed=calibration_seed,**shared):
        means.update(zip(piece.cell_ids,piece.mean_mjy));scatters.update(zip(piece.cell_ids,piece.scatter_mjy))
        calibrated+=piece.cell_ids.size
        if time.monotonic()-last_progress>=30 or calibrated==missing.size:
            print('PACS missing native calibrations completed',calibrated,'of',missing.size,flush=True)
            last_progress=time.monotonic()
    if resume_state is not None and 'table' in resume_state:
        table=resume_state['table'];lookup=resume_state.get('lookup')
        if lookup is None or lookup.shape!=(grid.n_cells,) or lookup.dtype!=np.int32:
            raise ValueError('Resume native table lacks matching cell lookup')
        if not np.array_equal(table.flux_grid,fluxgrid):
            raise ValueError('Resume curve grid differs')
        retained=lookup[lookup>=0]
        if not np.array_equal(np.sort(retained),np.arange(table.n_rows)):
            raise ValueError('Resume interrupted a native curve commit; reconcile its captured piece first')
        if table.arrays()['curves'].shape[0]!=table.n_rows:
            raise ValueError('Resume curve append was incomplete')
        if not np.array_equal(geometry['CURVE_ROW_'+band],lookup[cells]):
            raise ValueError('Resume geometry/curve lookup mismatch')
        done=lookup[cells]>=0
        for key in ('LOCAL_RMS_'+band+'_MJY','GAIN_'+band):
            values=geometry[key][done]
            if np.any(~np.isfinite(values)|(values<=0)):
                raise ValueError('Resume retained native row has incomplete '+key+' bookkeeping')
        for key in ('BACKGROUND_MEAN_'+band+'_MJY','CELL_SCATTER_'+band+'_MJY'):
            if np.any(~np.isfinite(geometry[key][done])):
                raise ValueError('Resume retained native row has incomplete '+key+' bookkeeping')
        nc=geometry['AREA_N_COVERED_'+band][done]
        ne=geometry['AREA_N_EXCLUDED_'+band][done]
        expected_excluded=np.divide(ne,nc,out=np.full(nc.size,np.nan),where=nc>0)
        if (np.any((nc<0)|(nc>area_draws)|(ne<0)|(ne>nc)) or
                not np.array_equal(geometry['AREA_COVERED_FRAC_'+band][done],nc/area_draws) or
                not np.array_equal(geometry['AREA_EXCLUDED_FRAC_'+band][done],expected_excluded,equal_nan=True)):
            raise ValueError('Resume retained native row has incomplete area bookkeeping')
    else:
        table=CurveTable(fluxgrid)
        lookup=np.full(grid.n_cells,-1,np.int32)
    pending=active[lookup[active]<0]
    completed=active.size-pending.size
    print('PACS native recovery retained',completed,'remaining',pending.size,
          'elapsed',time.monotonic()-finish_started,flush=True)
    last_recovery_progress=time.monotonic()
    for piece in iterate_cell_recoveries(config,record,cached_psf,pending,c.ra_deg[det],c.dec_deg[det],
                                         seed=curve_seed,**shared):
        cell_scatter=np.array([scatters[cell] for cell in piece.cell_ids])
        sigma=np.hypot(piece.pixel_sigma_mjy,cell_scatter[:,None])
        rows=table.append(piece.flux_mjy,sigma,piece.gain,
                          zero_flux_statistic=piece.flux_mjy/sigma)
        lookup[piece.cell_ids]=rows
        i=np.searchsorted(cells,piece.cell_ids)
        geometry['CURVE_ROW_'+band][i]=rows
        geometry['LOCAL_RMS_'+band+'_MJY'][i]=np.sqrt(np.mean(sigma*sigma,axis=1))
        geometry['GAIN_'+band][i]=piece.gain.mean(axis=1)
        geometry['BACKGROUND_MEAN_'+band+'_MJY'][i]=[means[cell] for cell in piece.cell_ids]
        geometry['CELL_SCATTER_'+band+'_MJY'][i]=cell_scatter
        geometry['AREA_N_COVERED_'+band][i]=piece.n_covered
        geometry['AREA_N_EXCLUDED_'+band][i]=piece.n_excluded
        geometry['AREA_COVERED_FRAC_'+band][i]=piece.n_covered/area_draws
        geometry['AREA_EXCLUDED_FRAC_'+band][i]=np.divide(piece.n_excluded,piece.n_covered,
            out=np.full(piece.cell_ids.size,np.nan),where=piece.n_covered>0)
        completed+=piece.cell_ids.size
        if time.monotonic()-last_recovery_progress>=30 or completed==active.size:
            print('PACS native curves',completed,'of',active.size,'elapsed',time.monotonic()-finish_started,flush=True)
            last_recovery_progress=time.monotonic()
    curves['SOURCE_ROW'][mapping[local],column]=lookup[singletons.cell_ids[local]]
    groups=[]
    for result in iter_group_targets(opened,singletons,exact_psf,half_size=half_size,
            annulus_arcsec=annulus_arcsec,min_coverage=min_coverage,
            calibration_seed=calibration_seed,curve_seed=curve_seed,area_draws=area_draws,
            max_draws=max_draws,n_positions=N_POS,beam_arcsec=beam):
        target_local,inseparable=apply_group_target(photo,mapping,result,band)
        target=result.target_source
        if len(inseparable):
            curves['SOURCE_ROW'][inseparable,column]=-1
            target_local=int(mapping[result.members[target]])
            row=-1
            s=result.curve_measurement.group_sigma[None,:]
            real_sigma=result.source_measurement.group_sigma[0]
            background_mean=result.calibration_mean_mjy.sum()
            gain=np.nan
        else:
            if target_local<0:
                continue
            f=result.curve_measurement.flux[:,target][None,:]
            s=result.curve_measurement.sigma[:,target][None,:]
            row=int(table.append(f,s,result.gain[None,:],zero_flux_statistic=f/s)[0])
            curves['SOURCE_ROW'][target_local,column]=row
            real_sigma=result.source_measurement.sigma[0,target]
            background_mean=result.calibration_mean_mjy[target]
            gain=float(result.gain.mean())
        groups.append({'GROUP_CURVE_ROW':row,'GROUP_ID':result.group_id,
            'GROUP_MAP_ID':record.map_id,
            'GROUP_TARGET_SOURCE_ROW':target_local,'GROUP_ANCHOR_CELL_ID':result.target_cell_id,
            'GROUP_N_MEMBER':len(result.members),'GROUP_REAL_SIGMA':real_sigma,
            'GROUP_RANDOM_SIGMA_MIN':float(s.min()),'GROUP_RANDOM_SIGMA_MAX':float(s.max()),
            'GROUP_GAIN':gain,'GROUP_BACKGROUND_MEAN':background_mean,
            'GROUP_AREA_N_CANDIDATE':int(result.area.n_candidate[0]),
            'GROUP_AREA_N_COVERED':int(result.area.n_covered[0]),
            'GROUP_AREA_N_EXCLUDED':int(result.area.n_excluded[0]),
            'members_source_row':c.source_row[result.members],
            'members_catalogue':c.catalogue[result.members],
            'covariance':result.calibration_covariance_mjy2})
    _put_group_geometry(geometry,band,groups)
    arrays=table.arrays()
    pexcluded=neighbours.excluded(pra,pdec,beam)
    prows=np.full(pcells.shape,-1,np.int32);prows[pcovered]=lookup[pcells[pcovered]]
    pixels=pixel_curves(arrays['curves'],prows,pcovered,pexcluded)
    curves.update({'CURVE_'+band:arrays['curves'],'MAG_GRID_'+band:magnitude,
        'N_POS_'+band:arrays['n_positions'],'COMPLETED_FROM_'+band:arrays['completed_from'],
        'ENVELOPE_'+band:arrays['envelope'],'PIXEL_CURVE_'+band:pixels.curves,
        'PIXEL_COVERED_FRAC_'+band:pixels.covered_fraction,
        'PIXEL_EXCLUDED_FRAC_'+band:pixels.excluded_fraction})
    for absent in (absent_bands if regional_complete else ()):
        ep,ec,eg=_empty_band(absent,names.size,admitted.size,magnitude)
        photo.update(ep);curves.update(ec);geometry.update(eg)
    print('PACS final curve audit',table.n_rows,'rows; max faint signal/noise',
          table.max_faint_signal_to_noise,'nonzero floor rows',np.count_nonzero(arrays['floor_counts']),flush=True)
    return write_products(config,region,photo,curves,geometry,publish=publish,
                          complete=regional_complete)


def build(config,regions=None,*,publish=False,inventory_complete=False,
          absent_bands=(),n_jobs=None,max_seconds=7200):
    """Run the authorized Musca map through calibration, groups and products.

    Current-map staging is useful before inventory closes, but cannot publish
    a regional product or fabricate an unprocessed band's absence. Additional
    selected maps require explicit ownership integration before this entry point
    can claim regional completion. No cross-region survey is started implicitly.
    """
    from .maps import read_index,open_map
    from .geometry import MapGrid,load_sesna_candidates
    from .psf import calibration_factory
    from .stage import run_singleton_fixed_point
    from .regional_workers import clear_worker_state
    selected=['Musca'] if regions is None else list(regions)
    if selected!=['Musca']:
        raise ValueError('This authorized execution builds Musca only')
    if publish and not inventory_complete:
        raise ValueError('Complete inventory is required before publishing')
    jobs=int(config.n_jobs if n_jobs is None else n_jobs)
    if jobs<1:
        raise ValueError('Positive worker count required')
    records=read_index(config,'Musca')
    if len(records)!=1:
        raise RuntimeError('Current Musca driver requires one selected logical map; additional-map ownership must be implemented')
    record=records[0]
    cached,exact=calibration_factory(config,record,half_size=12)
    try:
        with open_map(config,record) as opened:
            grid=MapGrid.from_wcs(opened.wcs,opened.image.shape)
            centre=grid.wcs.pixel_to_world((grid.shape[1]-1)/2,(grid.shape[0]-1)/2).icrs
            corners=grid.wcs.pixel_to_world(
                np.array([-.5,grid.shape[1]-.5,-.5,grid.shape[1]-.5]),
                np.array([-.5,-.5,grid.shape[0]-.5,grid.shape[0]-.5])).icrs
            radius=float(np.max(centre.separation(corners).deg)+P70_PARALLEL60_GROUP_RADIUS_ARCSEC/3600+1e-8)
            candidates,audit=load_sesna_candidates(config,centre.ra.deg,centre.dec.deg,radius)
            print('PACS complete map candidate audit',audit,flush=True)
            singletons=run_singleton_fixed_point(opened,candidates,cached,
                half_size=12,annulus_arcsec=(25.,35.),min_coverage=np.nextafter(0.,1.),
                seed=87421,area_draws=1024,max_draws=1048576,max_iterations=16,
                max_seconds=max_seconds,batch_size=N_POS,n_positions=N_POS,
                config=config,n_jobs=jobs)
            return finish_single_map(config,'Musca',record,opened,cached,exact,singletons,
                absent_bands=absent_bands,inventory_complete=inventory_complete,n_jobs=jobs,
                calibration_seed=87421,curve_seed=471027,publish=publish)
    finally:
        clear_worker_state()


def publish_staged(config,region,staged_paths,*,inventory_complete,absent_bands):
    """Publish validated current-map results after the archive inventory closes.

    Adds scientifically established absent-band arrays without redoing measured
    map work. Additional selected maps prevent promotion: their ownership and
    measurements must be integrated first. No coverage is inferred from a file
    merely being missing.
    """
    from .maps import read_index
    from .completeness import canonical_grid
    if not inventory_complete:
        raise ValueError('Completed archive inventory required for publication')
    if len(staged_paths)!=3:
        raise ValueError('Expected staged photometry, completeness and geometry paths')
    data=[]
    for path,quantity,granule in zip(staged_paths,('photometry','completeness','geometry'),('source','source','region')):
        path=Path(path)
        checked=checked_path(Path(config.data_root),str(path.relative_to(config.data_root)))
        prefix='.'+quantity+'_pacs_'+granule+'__'+region+'.hdf5.'
        if path!=checked or not path.name.startswith(prefix) or not path.name.endswith('.part'):
            raise ValueError('Staged product is not this region\'s checked PACS output')
        with h5py.File(path,'r') as handle:
            data.append({key:np.asarray(handle[key]) for key in handle})
    photo,curves,geometry=data
    records=read_index(config,region)
    if len(records)!=1 or 'CURVE_'+records[0].band not in curves:
        raise RuntimeError('Selected map inventory differs from staged single-map results')
    measured=[records[0].band]
    for band in set(BAND_KEYS)-set(measured):
        if 'CURVE_'+band in curves and (curves['CURVE_'+band].shape[0] or
                np.any(curves['PIXEL_COVERED_FRAC_'+band]!=0)):
            raise ValueError('Staged measurements include a band absent from selected map inventory')
    identities=np.asarray(geometry['MAP_ID_'+records[0].band]).astype('U')
    if identities.size and not np.all(identities==records[0].map_id):
        raise ValueError('Staged map identity differs from the selected archive map')
    if set(absent_bands)!=set(BAND_KEYS)-set(measured):
        raise ValueError('Explicit absent-band evidence must account for every unmeasured band')
    mag,_=canonical_grid()
    for band in absent_bands:
        pp,cc,gg=_empty_band(band,len(photo['NAME']),len(curves['HPX_PIX_512']),mag)
        photo.update(pp);curves.update(cc);geometry.update(gg)
    result=write_products(config,region,photo,curves,geometry,publish=True,complete=True)
    for path in staged_paths:
        Path(path).unlink()
    return result


def merge_staged_bands(config,region,band_products,*,inventory_complete,
                       absent_bands=(),publish=False):
    """Assemble independently measured band columns without discarding overlap.

    P100 is retained wherever observed even when P70 is also available. Inputs
    are already ownership-resolved regional products WITHIN each band; this
    function does not pretend that concatenating independent maps resolves
    same-band ownership or their shared detection/noise calibration.
    """
    from .maps import read_index
    from .completeness import canonical_grid
    if publish and not inventory_complete:
        raise ValueError('Regional publication requires completed inventory')
    if not band_products or not set(band_products)<=set(BAND_KEYS):
        raise ValueError('Supply staged products keyed by actual PACS band')
    if not set(absent_bands)<=set(BAND_KEYS) or set(absent_bands)&set(band_products):
        raise ValueError('Absent bands must be known bands disjoint from supplied measured products')
    records=read_index(config,region)
    observed={r.band for r in records}
    if set(absent_bands)&observed:
        raise ValueError('An indexed observed band cannot be declared absent')
    if inventory_complete and set(band_products)|set(absent_bands)!=set(BAND_KEYS):
        raise ValueError('Every band needs measured products or explicit absent-band evidence')
    photo={};curves={};geometry={}
    def owns(key,band):
        return key.endswith('_'+band) or ('_'+band+'_') in key
    for band in BAND_KEYS:
        if band not in band_products:
            continue
        paths=band_products[band]
        if len(paths)!=3:
            raise ValueError('Each band needs photometry/completeness/geometry staged paths')
        dictionaries=[]
        for path,quantity,granule in zip(paths,('photometry','completeness','geometry'),('source','source','region')):
            path=Path(path)
            checked=checked_path(Path(config.data_root),str(path.relative_to(config.data_root)))
            if (path!=checked or not path.name.startswith('.'+quantity+'_pacs_'+granule+'__'+region+'.hdf5.')
                    or not path.name.endswith('.part')):
                raise ValueError('Input is not a checked regional staged PACS product')
            with h5py.File(path,'r') as h:
                dictionaries.append({key:np.asarray(h[key]) for key in h})
        pp,cc,gg=dictionaries
        validate_products(pp,cc,gg,bands=(band,))
        expected_maps={r.map_id for r in records if r.band==band}
        actual_maps=set(np.asarray(gg['MAP_ID_'+band]).astype('U'))
        if inventory_complete and expected_maps!=actual_maps:
            raise ValueError('Band stage does not account for every selected logical map')
        if not photo:
            photo['NAME']=pp['NAME'];curves['HPX_PIX_512']=cc['HPX_PIX_512']
            curves['SOURCE_ROW']=np.full((len(pp['NAME']),len(BAND_KEYS)),-1,np.int32)
        elif (not np.array_equal(photo['NAME'],pp['NAME']) or
              not np.array_equal(curves['HPX_PIX_512'],cc['HPX_PIX_512'])):
            raise ValueError('Independent bands have different source/admitted-pixel order')
        photo.update({k:v for k,v in pp.items() if owns(k,band)})
        curves.update({k:v for k,v in cc.items() if owns(k,band)})
        geometry.update({k:v for k,v in gg.items() if owns(k,band)})
        column=BAND_KEYS.index(band)
        curves['SOURCE_ROW'][:,column]=cc['SOURCE_ROW'][:,column]
    mag,_=canonical_grid()
    for band in absent_bands:
        pp,cc,gg=_empty_band(band,len(photo['NAME']),len(curves['HPX_PIX_512']),mag)
        photo.update(pp);curves.update(cc);geometry.update(gg)
    complete=inventory_complete and set(band_products)|set(absent_bands)==set(BAND_KEYS)
    return write_products(config,region,photo,curves,geometry,publish=publish,complete=complete)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Musca PACS calibrated photometry and completeness')
    parser.add_argument('config')
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--inventory-complete',action='store_true')
    parser.add_argument('--absent-band',choices=BAND_KEYS,action='append',default=[])
    parser.add_argument('--jobs',type=int)
    args=parser.parse_args()
    build(config_module.load(args.config),publish=args.publish,
          inventory_complete=args.inventory_complete,absent_bands=args.absent_band,n_jobs=args.jobs)
