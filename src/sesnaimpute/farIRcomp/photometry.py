"""Bounded real-map cutout adapter shared by extraction and injection checks.

This module provides paired-source measurements and classification, without a
production writer. Undetected SESNA neighbours do not exclude sky or trigger
groups. First-pass detected PACS neighbours define grouping and exclusion;
unseparable groups store bounds for future use while the band is excluded today.
"""
from dataclasses import dataclass
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from .extractor import prepare_fit, measure_sources
from .constants import (DETECTION_SNR,ORIGIN_DETECTED,ORIGIN_UNDETECTED,
                        ORIGIN_NOT_COVERED,ORIGIN_UPPER_BOUND,UPPER_LIMIT_SIGMA,
                        PSF_INTEGRATION_ORDER)


def first_pass_detected(measurements, covered):
    """Freeze detections from calibrated singleton physical-source fits.

    Invalid covered fits are errors, never nondetections or silent exclusions.
    This mask is shared by grouping and completeness; later joint fits do not
    change it. Caller must have calibrated/frozen the background scatter first.
    """
    covered = np.asarray(covered,dtype=bool)
    if measurements.flux.shape != (covered.size,1) or covered.ndim != 1:
        raise ValueError('First pass requires one physical source per covered flag')
    good = (measurements.valid & measurements.identifiable[:,0] &
            np.isfinite(measurements.flux[:,0]) & np.isfinite(measurements.sigma[:,0]) &
            (measurements.sigma[:,0] > 0))
    if np.any(covered & ~good):
        raise ValueError('A covered singleton has an invalid calibrated measurement')
    detected = covered & good & (measurements.flux[:,0] >= DETECTION_SNR*measurements.sigma[:,0])
    detected.flags.writeable = False
    return detected


@dataclass(frozen=True)
class SourceClassification:
    origin: np.ndarray
    source_row: np.ndarray
    n_blend: np.ndarray
    group_id: np.ndarray
    forced_flux_mjy: np.ndarray
    forced_sigma_mjy: np.ndarray
    stored_bound_mjy: np.ndarray
    upper_limit_sigma: np.ndarray
    requires_group_curve: np.ndarray
    ready: np.ndarray
    condition: np.ndarray


def classify_measurements(measurements, covered, curve_rows, *, group_ids,
                          group_curve_rows=None):
    """Classify supplied singleton or joint physical-source measurements.

    Arrays have shape (n_fit,n_physical_source); each fit is a complete group.
    Multi-source inputs must be groups formed from the frozen first-pass graph.
    No heuristic correlation threshold is added: physical-sum identifiability
    and valid numerical covariance determine whether a group is separable.

    A separable grouped nondetection requires a curve measured with that same
    joint extractor. Until its row is supplied, ready=False and source_row=-1
    prevent accidental use of a singleton curve; callers must not publish it.
    Unseparable groups store a future bound but exclude the band today. The
    dimensionless upper-limit SIGMA sentinel is separate from mJy errors.
    """
    flux,sigma = measurements.flux,measurements.sigma
    covered = np.asarray(covered,dtype=bool)
    rows = np.asarray(curve_rows)
    if covered.shape != flux.shape or rows.shape != flux.shape or not np.issubdtype(rows.dtype,np.integer):
        raise ValueError('Coverage and integer curve rows must match physical-source arrays')
    nfit,nsource = flux.shape
    if nsource>1 and np.any(np.any(covered,axis=1)&~np.all(covered,axis=1)):
        raise ValueError('Partly covered groups need explicit tile/member handling before classification')
    ids = np.asarray(group_ids)
    if ids.shape != (nfit,) or not np.issubdtype(ids.dtype,np.integer):
        raise ValueError('One integer group ID is required per fit')
    numeric = measurements.valid & np.all(np.isfinite(measurements.pixel_covariance),axis=(1,2))
    if np.any(np.any(covered,axis=1)&~numeric):
        raise ValueError('Invalid covered joint fit; cannot turn failure into a stored bound')
    member_good = measurements.identifiable & np.isfinite(flux) & np.isfinite(sigma) & (sigma>0)
    separable = np.all(member_good,axis=1)
    unresolved = np.any(covered,axis=1)&~separable
    if np.any(unresolved & (nsource==1)):
        raise ValueError('Invalid singleton cannot be classified as an unseparable detected group')
    if np.any(unresolved & (~measurements.group_identifiable | ~np.isfinite(measurements.group_sigma) |
                            (measurements.group_sigma<=0) | ~np.isfinite(measurements.group_total))):
        raise ValueError('Unseparable group needs identifiable total and full background covariance')
    detected = member_good & (flux >= DETECTION_SNR*sigma)
    origin = np.where(covered,np.where(detected,ORIGIN_DETECTED,ORIGIN_UNDETECTED),ORIGIN_NOT_COVERED)
    source_row = np.where(covered,rows,-1).astype(np.int64)
    bound = np.full(flux.shape,np.nan)
    sentinel = np.full(flux.shape,np.nan)
    origin[unresolved] = ORIGIN_UPPER_BOUND
    source_row[unresolved] = -1
    bound[unresolved] = (np.maximum(0,measurements.group_total[unresolved])+
                         DETECTION_SNR*measurements.group_sigma[unresolved])[:,None]
    sentinel[unresolved] = UPPER_LIMIT_SIGMA
    needs = (origin==ORIGIN_UNDETECTED)&(nsource>1)
    if group_curve_rows is None:
        source_row[needs] = -1
    else:
        joint_rows = np.asarray(group_curve_rows)
        if joint_rows.shape != flux.shape or not np.issubdtype(joint_rows.dtype,np.integer):
            raise ValueError('Group-matched curve rows must match source arrays')
        source_row[needs] = joint_rows[needs]
    ready = ~((origin==ORIGIN_UNDETECTED)&(source_row<0))
    return SourceClassification(origin,source_row,np.full(flux.shape,nsource,dtype=np.int32),
                                np.broadcast_to(ids[:,None],flux.shape),flux,sigma,bound,sentinel,
                                needs&(source_row<0),ready,measurements.condition)


@dataclass(frozen=True)
class CutoutBatch:
    image: np.ndarray
    error: np.ndarray
    coverage: np.ndarray
    templates: np.ndarray
    annulus_mask: np.ndarray
    covered: np.ndarray
    min_coverage: float

    def measure(self, *, cell_scatter_mjy):
        context = prepare_fit(self.image, self.error, self.coverage, self.templates,
                              self.annulus_mask, min_coverage=self.min_coverage)
        return measure_sources(self.image,context,self.templates,np.array([0,0]),
                               cell_scatter_mjy=cell_scatter_mjy)


def _geometry(open_map, ra_deg, dec_deg, half_size):
    if not isinstance(half_size, int) or half_size < 1:
        raise ValueError('half_size must be a positive integer')
    ra, dec = np.broadcast_arrays(np.asarray(ra_deg, float), np.asarray(dec_deg, float))
    if ra.ndim != 1 or np.any(~np.isfinite(ra)) or np.any(~np.isfinite(dec)):
        raise ValueError('Positions must be finite 1D arrays')
    sky = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs').transform_to(
        open_map.wcs.pixel_to_world(0, 0).frame)
    x, y = open_map.wcs.world_to_pixel(sky)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError('Position outside WCS projection')
    d = np.arange(-half_size, half_size+1)
    yy, xx = np.meshgrid(d, d, indexing='ij')
    px = np.floor(x+.5).astype(np.int64)[:,None,None]+xx
    py = np.floor(y+.5).astype(np.int64)[:,None,None]+yy
    ny, nx = open_map.image.shape
    inside = (px >= 0)&(px < nx)&(py >= 0)&(py < ny)
    return sky, px, py, inside


def positions_covered(open_map, ra_deg, dec_deg, *, half_size, min_coverage):
    """Usable centre pixel plus in-map cutout, fixed before injection.

    Invalid surrounding pixels remain for the extractor's mask. The complete
    stencil must be inside the map; distinguish this pilot geometry from raw
    coverage area. This is not a detection or detected-PACS exclusion mask.
    """
    if not np.isfinite(min_coverage) or min_coverage <= 0:
        raise ValueError('min_coverage must be positive and finite')
    if not isinstance(half_size,int) or half_size < 1:
        raise ValueError('half_size must be a positive integer')
    ra,dec = np.broadcast_arrays(np.asarray(ra_deg,float),np.asarray(dec_deg,float))
    if ra.ndim != 1 or np.any(~np.isfinite(ra)) or np.any(~np.isfinite(dec)):
        raise ValueError('Positions must be finite 1D arrays')
    sky = SkyCoord(ra*u.deg,dec*u.deg,frame='icrs')
    x,y = open_map.wcs.world_to_pixel(sky)
    finite = np.isfinite(x)&np.isfinite(y)
    x = np.floor(np.where(finite,x,0)+.5).astype(np.int64)
    y = np.floor(np.where(finite,y,0)+.5).astype(np.int64)
    ny, nx = open_map.image.shape
    inside = finite & (x >= half_size)&(x < nx-half_size)&(y >= half_size)&(y < ny-half_size)
    xx, yy = np.clip(x,0,nx-1), np.clip(y,0,ny-1)
    image, error, cov = [a[yy,xx] for a in (open_map.image,open_map.error,open_map.coverage)]
    valid = np.isfinite(image) & np.isfinite(error) & (error > 0) & np.isfinite(cov) & (cov >= min_coverage)
    return inside & valid


def prepare_positions(open_map, ra_deg, dec_deg, psf, *, half_size,
                      annulus_arcsec, min_coverage,
                      integration_order=PSF_INTEGRATION_ORDER):
    """Copy only small cutouts; convert Jy/pixel image and sigma to mJy/pixel.

    Annulus radii and fit extent are explicit diagnostic geometry, not calibrated
    PACS constants. No fitted results are used to select or discard positions.
    """
    if open_map.record.unit != 'Jy/pixel' or open_map.record.error_unit != 'Jy/pixel':
        raise ValueError('Adapter requires matching Jy/pixel science and sigma layers')
    inner, outer = annulus_arcsec
    if not np.isfinite(inner+outer) or not 0 < inner < outer:
        raise ValueError('Annulus requires finite 0 < inner < outer')
    sky, x, y, inside = _geometry(open_map, ra_deg, dec_deg, half_size)
    covered = positions_covered(open_map,ra_deg,dec_deg,half_size=half_size,
                                min_coverage=min_coverage)
    if not np.all(inside):
        raise ValueError('Prepare only complete in-map cutouts; sample coverage first')
    image = np.array(open_map.image[y,x], dtype=float)*1000
    error = np.array(open_map.error[y,x], dtype=float)*1000
    coverage = np.array(open_map.coverage[y,x], dtype=float)
    source = sky[:,None,None]
    templates = psf(open_map.wcs,source,x,y,integration_order)
    if templates.ndim == 3:  # Explicit diagnostic mixture; not final paired extractor.
        templates = templates[:,None]
    radius = source.separation(open_map.wcs.pixel_to_world(x,y)).arcsec
    annulus = (radius >= inner)&(radius <= outer)
    if np.any(~np.any(annulus,axis=(1,2))):
        raise ValueError('Annulus has no pixels in cutout')
    return CutoutBatch(image,error,coverage,templates,annulus,covered,float(min_coverage))


def prepare_xy(open_map, source_x, source_y, psf, *, half_size, annulus_arcsec,
               min_coverage, integration_order=PSF_INTEGRATION_ORDER):
    """Native-pixel batch adapter, optionally using a validated PSF cache."""
    x,y = np.broadcast_arrays(np.asarray(source_x,float),np.asarray(source_y,float))
    if x.ndim != 1 or np.any(~np.isfinite(x+y)) or not isinstance(half_size,int) or half_size<1:
        raise ValueError('Finite 1D positions and positive integer half_size required')
    if open_map.record.unit!='Jy/pixel' or open_map.record.error_unit!='Jy/pixel':
        raise ValueError('Matching Jy/pixel science/error layers required')
    inner,outer=annulus_arcsec
    if not np.isfinite(inner+outer) or not 0<inner<outer:
        raise ValueError('Invalid annulus')
    d=np.arange(-half_size,half_size+1)
    yy,xx=np.meshgrid(d,d,indexing='ij')
    px=np.floor(x+.5).astype(np.int64)[:,None,None]+xx
    py=np.floor(y+.5).astype(np.int64)[:,None,None]+yy
    ny,nx=open_map.image.shape
    if np.any((px<0)|(px>=nx)|(py<0)|(py>=ny)):
        raise ValueError('Cutout exceeds map; determine coverage before extraction')
    image=np.asarray(open_map.image[py,px],float)*1000
    error=np.asarray(open_map.error[py,px],float)*1000
    coverage=np.asarray(open_map.coverage[py,px],float)
    if hasattr(psf,'evaluate_xy'):
        templates=psf.evaluate_xy(x,y,px,py)
    else:
        sky=open_map.wcs.pixel_to_world(x,y)[:,None,None]
        templates=psf(open_map.wcs,sky,px,py,integration_order)
    annulus=_native_annulus_mask(open_map.wcs,x,y,px,py,inner,outer)
    h=half_size
    centre=(np.isfinite(image[:,h,h])&np.isfinite(error[:,h,h])&(error[:,h,h]>0)&
            np.isfinite(coverage[:,h,h])&(coverage[:,h,h]>=min_coverage))
    return CutoutBatch(image,error,coverage,templates,annulus,centre,float(min_coverage))


def prepare_group_xy(open_map, source_x, source_y, psf, *, half_size,
                     annulus_arcsec, min_coverage,
                     integration_order=PSF_INTEGRATION_ORDER):
    """Joint paired design on the union of the members' singleton supports.

    Positions have shape (trial,physical_source). Every source contributes both
    released scan columns on every fitted pixel, including its wings in other
    members' stamps. No source groups or templates are silently split/truncated.
    Use the exact PairedPSF here when a singleton response cache is too narrow.
    """
    x,y=np.broadcast_arrays(np.asarray(source_x,float),np.asarray(source_y,float))
    if x.ndim!=2 or x.shape[1]<1 or np.any(~np.isfinite(x+y)):
        raise ValueError('Finite (trial,source) coordinates required')
    if not isinstance(half_size,int) or half_size<1:
        raise ValueError('Positive integer half_size required')
    if open_map.record.unit!='Jy/pixel' or open_map.record.error_unit!='Jy/pixel':
        raise ValueError('Matching Jy/pixel science/error layers required')
    inner,outer=annulus_arcsec
    if not 0<inner<outer:
        raise ValueError('Invalid annulus radii')
    ix,iy=np.floor(x+.5).astype(np.int64),np.floor(y+.5).astype(np.int64)
    left,bottom=ix.min(axis=1)-half_size,iy.min(axis=1)-half_size
    width=int(np.max(ix.max(axis=1)+half_size-left))+1
    height=int(np.max(iy.max(axis=1)+half_size-bottom))+1
    if x.shape[0]*width*height*(2*x.shape[1]+5)*8 > 512*1024**2:
        raise ValueError('Joint stamp exceeds 512 MiB work budget; reduce trial batch')
    dy,dx=np.mgrid[:height,:width]
    px=left[:,None,None]+dx;py=bottom[:,None,None]+dy
    ny,nx=open_map.image.shape
    inside=(px>=0)&(px<nx)&(py>=0)&(py<ny)
    if not np.all(inside):
        raise ValueError('Joint stamp exceeds map; establish group coverage first')
    image=np.asarray(open_map.image[py,px],float)*1000
    error=np.asarray(open_map.error[py,px],float)*1000
    coverage=np.asarray(open_map.coverage[py,px],float).copy()
    pixel_sky=open_map.wcs.pixel_to_world(px,py)
    support=np.zeros(px.shape,bool);fit_support=np.zeros(px.shape,bool)
    templates=[]
    for j in range(x.shape[1]):
        own=(np.abs(px-ix[:,j,None,None])<=half_size)&(np.abs(py-iy[:,j,None,None])<=half_size)
        sky=open_map.wcs.pixel_to_world(x[:,j],y[:,j])[:,None,None]
        radius=sky.separation(pixel_sky).arcsec
        ring=(radius>=inner)&(radius<=outer)
        support|=own;fit_support|=own&~ring
        if hasattr(psf,'evaluate_xy'):
            response=psf.evaluate_xy(x[:,j],y[:,j],px,py)
        else:
            response=psf(open_map.wcs,sky,px,py,integration_order)
        templates.append(response)
    coverage[~support]=0.
    centre=(np.isfinite(open_map.image[iy,ix])&np.isfinite(open_map.error[iy,ix])&
            (open_map.error[iy,ix]>0)&np.isfinite(open_map.coverage[iy,ix])&
            (open_map.coverage[iy,ix]>=min_coverage))
    return CutoutBatch(image,error,coverage,np.concatenate(templates,axis=1),
                       support&~fit_support,np.all(centre,axis=1),float(min_coverage))


def measure_group_xy(open_map, source_x, source_y, psf, *, half_size,
                     annulus_arcsec, min_coverage, cell_covariance_mjy2):
    """Joint physical sums with an explicit frozen full sky covariance."""
    batch=prepare_group_xy(open_map,source_x,source_y,psf,half_size=half_size,
                           annulus_arcsec=annulus_arcsec,min_coverage=min_coverage)
    context=prepare_fit(batch.image,batch.error,batch.coverage,batch.templates,
                        batch.annulus_mask,min_coverage=min_coverage)
    nsource=batch.templates.shape[1]//2
    cov=np.asarray(cell_covariance_mjy2,float)
    scatter=np.sqrt(np.maximum(np.diagonal(cov,axis1=-2,axis2=-1),0))
    measurement=measure_sources(batch.image,context,batch.templates,
                                np.repeat(np.arange(nsource),2),
                                cell_scatter_mjy=scatter,cell_covariance_mjy2=cov)
    if not np.all(batch.covered & measurement.valid & measurement.group_identifiable):
        raise RuntimeError('Invalid accepted joint fit; no silent rejection')
    return measurement,batch,context


def source_product_arrays(singletons, region_catalogue_filename, n_source, record):
    """Prepare one band's source arrays in the original regional row order.

    EXTENDED is unassessed: the writer must label this explicitly in READING;
    these False values do not assert a successful morphology measurement.
    Mapping is -1 for nuisance candidates belonging to another catalogue.
    """
    c=singletons.candidates
    mapping=np.full(c.ra_deg.size,-1,dtype=np.int64)
    keep=np.asarray(c.catalogue)==str(region_catalogue_filename)
    local=np.asarray(c.source_row[keep],dtype=np.int64)
    if np.any((local<0)|(local>=n_source)) or np.unique(local).size!=local.size:
        raise ValueError('Regional candidate rows must be unique original catalogue indices')
    mapping[keep]=local
    band=record.band
    obs=','.join(str(v) for v in record.obsids).encode('ascii')
    arrays={'FNU_'+band+'_MJY':np.full(n_source,np.nan,dtype=np.float64),
            'SIGMA_'+band+'_MJY':np.full(n_source,np.nan,dtype=np.float64),
            'ORIGIN_'+band:np.full(n_source,ORIGIN_NOT_COVERED,dtype=np.int8),
            'FIRST_PASS_DETECTED_'+band:np.zeros(n_source,dtype=np.bool_),
            'LOCAL_RMS_'+band+'_MJY':np.full(n_source,np.nan,dtype=np.float32),
            'OBSID_'+band:np.full(n_source,b'',dtype='S'+str(max(1,len(obs)))),
            'N_BLEND_'+band:np.zeros(n_source,dtype=np.int16),
            'BLEND_GROUP_'+band:np.full(n_source,-1,dtype=np.int32),
            'EXTENDED_'+band:np.zeros(n_source,dtype=np.bool_)}
    selected=np.flatnonzero(keep&singletons.covered)
    rows=mapping[selected]
    arrays['FNU_'+band+'_MJY'][rows]=singletons.flux_mjy[selected]
    arrays['SIGMA_'+band+'_MJY'][rows]=singletons.sigma_mjy[selected]
    arrays['LOCAL_RMS_'+band+'_MJY'][rows]=singletons.sigma_mjy[selected]
    arrays['ORIGIN_'+band][rows]=np.where(singletons.detected[selected],ORIGIN_DETECTED,ORIGIN_UNDETECTED)
    arrays['OBSID_'+band][rows]=obs
    arrays['N_BLEND_'+band][rows]=1
    arrays['FIRST_PASS_DETECTED_'+band][rows]=singletons.detected[selected]
    return arrays,mapping


def apply_group_target(arrays,candidate_to_local,result,band):
    """Apply a matched target result; return rows needing a curve or exclusion.

    Returns (separable_target_local_row, inseparable_local_rows). The first is
    -1 for a nuisance target outside this catalogue. Inseparable groups use the
    canonical member-zero anchor's full covariance and store the future bound
    max(0,total)+5*sigma; all their current likelihood rows must become -1.
    """
    measurement=result.source_measurement
    if not measurement.valid[0] or not measurement.group_identifiable[0]:
        raise ValueError('Invalid actual group cannot enter a science product')
    members=np.asarray(result.members,dtype=np.int64)
    if members.size>np.iinfo(np.int16).max:
        raise ValueError('Physical group size exceeds N_BLEND storage')
    local=np.asarray(candidate_to_local)[members]
    known=local[local>=0]
    arrays['N_BLEND_'+band][known]=members.size
    arrays['BLEND_GROUP_'+band][known]=result.group_id
    if not np.all(measurement.identifiable[0]):
        if result.target_source!=0:
            raise ValueError('Inseparable bound requires the canonical member-zero calibration')
        sigma=float(measurement.group_sigma[0]);flux=float(measurement.group_total[0])
        if not np.isfinite(sigma+flux) or sigma<=0:
            raise ValueError('Inseparable group requires a finite full-covariance total error')
        arrays['FNU_'+band+'_MJY'][known]=max(0.,flux)+DETECTION_SNR*sigma
        arrays['SIGMA_'+band+'_MJY'][known]=UPPER_LIMIT_SIGMA
        arrays['LOCAL_RMS_'+band+'_MJY'][known]=sigma
        arrays['ORIGIN_'+band][known]=ORIGIN_UPPER_BOUND
        return -1,known
    target=int(result.target_source);row=int(local[target])
    flux=float(measurement.flux[0,target]);sigma=float(measurement.sigma[0,target])
    if not np.isfinite(flux+sigma) or sigma<=0:
        raise ValueError('Separable physical source has invalid calibrated uncertainty')
    if row>=0:
        arrays['FNU_'+band+'_MJY'][row]=flux
        arrays['SIGMA_'+band+'_MJY'][row]=sigma
        arrays['LOCAL_RMS_'+band+'_MJY'][row]=sigma
        arrays['ORIGIN_'+band][row]=ORIGIN_DETECTED if flux>=DETECTION_SNR*sigma else ORIGIN_UNDETECTED
    return row,np.empty(0,dtype=np.int64)


def _native_annulus_mask(wcs, source_x, source_y, pixel_x, pixel_y, inner, outer):
    """Exact TAN ray geometry, preserving legacy masks at rounding boundaries.

    TAN rays are proportional to (u,v,1) in an orthonormal tangent frame.
    Their angle is atan2(norm(cross),dot); sky-frame rotation cancels. Unsupported
    projections/distortions use the original Astropy calculation. Within 64
    float64 epsilon radians of either boundary, redo that entire stamp with
    Astropy, preserving its inclusive-boundary convention at machine precision.
    """
    x,y=np.asarray(source_x,float),np.asarray(source_y,float)
    px,py=np.broadcast_arrays(np.asarray(pixel_x,float),np.asarray(pixel_y,float))
    def reference(rows):
        sky=wcs.pixel_to_world(x[rows],y[rows])[:,None,None]
        r=sky.separation(wcs.pixel_to_world(px[rows],py[rows])).arcsec
        return (r>=inner)&(r<=outer)
    supported=(not wcs.has_distortion and
               tuple(wcs.wcs.ctype)==('RA---TAN','DEC--TAN') and
               not wcs.wcs.get_pv() and
               all(str(unit)=='deg' for unit in wcs.wcs.cunit))
    if not supported:
        return reference(np.arange(x.size))
    matrix=np.deg2rad(wcs.pixel_scale_matrix)
    ref=wcs.wcs.crpix-1
    dx,dy=x-ref[0],y-ref[1]
    us=(matrix[0,0]*dx+matrix[0,1]*dy)[:,None,None]
    vs=(matrix[1,0]*dx+matrix[1,1]*dy)[:,None,None]
    dx,dy=px-ref[0],py-ref[1]
    up=matrix[0,0]*dx+matrix[0,1]*dy
    vp=matrix[1,0]*dx+matrix[1,1]*dy
    cross=np.sqrt((vs-vp)**2+(up-us)**2+(us*vp-vs*up)**2)
    radius=np.rad2deg(np.arctan2(cross,1+us*up+vs*vp))*3600
    mask=(radius>=inner)&(radius<=outer)
    guard=np.rad2deg(64*np.finfo(np.float64).eps)*3600
    ambiguous=np.any((np.abs(radius-inner)<=guard)|(np.abs(radius-outer)<=guard),axis=(1,2))
    if np.any(ambiguous):
        rows=np.flatnonzero(ambiguous)
        mask[rows]=reference(rows)
    return mask
