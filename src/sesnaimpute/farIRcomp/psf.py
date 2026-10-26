"""Published PACS P70 parallel-60 templates for controlled injection diagnostics.

The supplied components are NOT automatically a calibrated PSF for a combined
map: local scan weights and projection compatibility require separate evidence.
PICC-ME-TN-033 v2.2, sections 4, 8, 9 and appendix define the orientation and
normalization. https://archives.esac.esa.int/hsa/legacy/ADP/PSF/PACS/PACS-P/
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class PublishedPSF:
    image: np.ndarray
    path: str
    position_angle_deg: float

    @classmethod
    def read_p70_parallel60(cls, path, *, position_angle_deg):
        """Read a normal-processing OD345 +/-42 pixfrac=.1 release component.

        Release images have unit flux inside 60 arcsec, with background removed
        in 60--70 arcsec. Multiply by .911 to use the released total-flux
        convention (~1000 arcsec); never normalize the cropped image sum.
        EEF_blu_60para_21.txt gives E(r)=.911 V(r)+2.18805296634e-6*pi*r².
        The removed constant pedestal is absorbed by the fitted background.
        """
        path = Path(path)
        allowed = ['psf20_blu_60para_vesta_od345_ama' + a + '_0p10.fits.gz'
                   for a in ('+42', '-42')]
        if path.name not in allowed or not np.isfinite(position_angle_deg):
            raise ValueError('Require the named official normal-processing P70 templates and finite PA')
        with fits.open(path) as hdus:
            h = hdus['image'].header
            data = np.array(hdus['image'].data, dtype=float)
            if (data.shape != (201, 201) or not np.all(np.isfinite(data)) or
                    not np.allclose([h['CDELT1'], h['CDELT2']], [-1/3600, 1/3600]) or
                    (h['CRPIX1'], h['CRPIX2']) != (101., 101.)):
                raise ValueError('Unexpected published PSF geometry')
            yy, xx = np.indices(data.shape)
            if not np.isclose(data[np.hypot(xx-100, yy-100) <= 60].sum(), 1., atol=2e-4):
                raise ValueError('Release 60 arcsec normalization not satisfied')
        data *= .911
        data.flags.writeable = False
        return cls(data, str(path), float(position_angle_deg))

    def density(self, east_arcsec, north_arcsec):
        """Flux per square arcsec; +Z is up, template x increases westward."""
        a = np.deg2rad(self.position_angle_deg)
        east0 = east_arcsec*np.cos(a) - north_arcsec*np.sin(a)
        north0 = east_arcsec*np.sin(a) + north_arcsec*np.cos(a)
        x, y = 100-east0, 100+north0
        if np.any((x < 0) | (x > 200) | (y < 0) | (y > 200)):
            raise ValueError('Requested support exceeds published PSF; wings must not be invented')
        return map_coordinates(self.image, [y, x], order=1, mode='nearest', prefilter=False)


@dataclass(frozen=True)
class DiagnosticMixture:
    """Explicit, fixed component weights; no default or calibrated-map claim."""
    components: tuple
    weights: tuple

    def __post_init__(self):
        w = np.asarray(self.weights)
        if (not self.components or len(w) != len(self.components) or
                np.any(~np.isfinite(w)) or np.any(w < 0) or not np.isclose(w.sum(), 1.)):
            raise ValueError('Explicit nonnegative component weights must sum to one')

    def __call__(self, wcs, source, pixel_x, pixel_y, integration_order, *, _separate=False):
        """Gauss-Legendre pixel integration, preserving finite PSF normalization.

        Pixel area is computed from spherical corner coordinates locally. This
        interpolates the released 1-arcsec pixel-convolved PSF; it is a diagnostic
        resampling, not an exact reconstruction of the mapper's timeline drizzle.
        Quadrature sky coordinates interpolate exact pixel-corner offsets. On
        the Musca 3.2-arcsec TAN grid this approximation was checked at the centre
        and four corners: maximum error <4e-7 arcsec over a 25x25 stencil.
        Other map geometries require the same independent accuracy check.
        """
        if not isinstance(integration_order, int) or integration_order < 1:
            raise ValueError('integration_order must be a positive integer')
        nodes, weights = np.polynomial.legendre.leggauss(integration_order)
        corners = [wcs.pixel_to_world(pixel_x+dx, pixel_y+dy)
                   for dx, dy in ((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5))]
        offsets = [source.spherical_offsets_to(c) for c in corners]
        east = np.array([v[0].arcsec for v in offsets])
        north = np.array([v[1].arcsec for v in offsets])
        area = .5*np.abs(np.sum(east*np.roll(north,-1,axis=0)-north*np.roll(east,-1,axis=0),axis=0))
        out = np.zeros((len(self.components),*pixel_x.shape),dtype=float) if _separate else np.zeros_like(pixel_x,dtype=float)
        for u, wu in zip(nodes/2, weights/2):
            for v, wv in zip(nodes/2, weights/2):
                coefficients = ((.5-u)*(.5-v),(.5+u)*(.5-v),
                                (.5+u)*(.5+v),(.5-u)*(.5+v))
                e = sum(c*value for c,value in zip(coefficients,east))
                n = sum(c*value for c,value in zip(coefficients,north))
                for index,(component, fraction) in enumerate(zip(self.components, self.weights)):
                    if _separate:
                        out[index] += wu*wv*component.density(e,n)
                    elif fraction:
                        out += wu*wv*fraction*component.density(e, n)
        return np.moveaxis(out*area,0,1) if _separate else out*area


@dataclass(frozen=True)
class _ReferencePairedPSF:
    """Two unconstrained scan columns per physical source, sharing quadrature.

    Their SUM is the flux estimator. No assumed scan-weight mixture is imposed.
    This spans mixture uncertainty, not arbitrary PSF or mapper differences.
    """
    components: tuple

    def __post_init__(self):
        if len(self.components) != 2:
            raise ValueError('Exactly two published scan components required')

    def __call__(self,wcs,source,pixel_x,pixel_y,integration_order):
        if pixel_x.ndim != 3:
            raise ValueError('PairedPSF expects a batch of 2D cutouts')
        return DiagnosticMixture(self.components,(.5,.5))(
            wcs,source,pixel_x,pixel_y,integration_order,_separate=True)


from numba import njit


@njit(cache=True)
def _paired_quadrature(east, north, images, cosine, sine, nodes, weights):
    """Fused pixel integration; no quadrature-sized temporary image arrays."""
    n_position, n_pixel = east.shape[1:]
    out = np.zeros((n_position, images.shape[0], n_pixel), dtype=np.float64)
    for row in range(n_position):
        for pixel in range(n_pixel):
            area = 0.0
            for corner in range(4):
                nxt = (corner + 1) % 4
                area += (east[corner,row,pixel]*north[nxt,row,pixel]
                         - north[corner,row,pixel]*east[nxt,row,pixel])
            area = 0.5 * abs(area)
            for iu in range(nodes.size):
                u = nodes[iu]
                for iv in range(nodes.size):
                    v = nodes[iv]
                    coeff = ((.5-u)*(.5-v), (.5+u)*(.5-v),
                             (.5+u)*(.5+v), (.5-u)*(.5+v))
                    e, n = 0.0, 0.0
                    for corner in range(4):
                        e += coeff[corner]*east[corner,row,pixel]
                        n += coeff[corner]*north[corner,row,pixel]
                    weight = weights[iu]*weights[iv]*area
                    for member in range(images.shape[0]):
                        x = 100.0 - (e*cosine[member] - n*sine[member])
                        y = 100.0 + (e*sine[member] + n*cosine[member])
                        if x < 0 or x > 200 or y < 0 or y > 200:
                            raise ValueError('Requested support exceeds published PSF')
                        ix, iy = min(int(x),199), min(int(y),199)
                        fx, fy = x-ix, y-iy
                        value = ((1-fx)*(1-fy)*images[member,iy,ix]
                                 + fx*(1-fy)*images[member,iy,ix+1]
                                 + fx*fy*images[member,iy+1,ix+1]
                                 + (1-fx)*fy*images[member,iy+1,ix])
                        out[row,member,pixel] += weight*value
    return out


@dataclass(frozen=True)
class PairedPSF:
    """Separate published component columns with shared quadrature geometry.

    Output is (n_position,n_component,ny,nx), without mixture weights or
    renormalization. Uses the same corner interpolation as DiagnosticMixture;
    independent exact-WCS validation is required for a new map geometry.
    """
    components: tuple

    def __post_init__(self):
        if len(self.components) != 2 or not all(isinstance(c, PublishedPSF)
                                               for c in self.components):
            raise ValueError('PairedPSF requires two published components')

    def __call__(self, wcs, source, pixel_x, pixel_y, integration_order):
        if not isinstance(integration_order,int) or integration_order < 1:
            raise ValueError('integration_order must be a positive integer')
        x, y = np.broadcast_arrays(np.asarray(pixel_x,float),np.asarray(pixel_y,float))
        if x.ndim != 3 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError('Pixel coordinates must be finite (n_position,ny,nx) arrays')
        east, north = [], []
        for dx, dy in ((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)):
            e, n = source.spherical_offsets_to(wcs.pixel_to_world(x+dx,y+dy))
            east.append(e.arcsec);north.append(n.arcsec)
        shape=(4,x.shape[0],x.shape[1]*x.shape[2])
        east=np.ascontiguousarray(east).reshape(shape)
        north=np.ascontiguousarray(north).reshape(shape)
        nodes, weights=np.polynomial.legendre.leggauss(integration_order)
        angles=np.deg2rad([c.position_angle_deg for c in self.components])
        images=np.stack([c.image for c in self.components])
        out=_paired_quadrature(east,north,images,np.cos(angles),np.sin(angles),
                               nodes/2,weights/2)
        return out.reshape((x.shape[0],2,x.shape[1],x.shape[2]))


@njit(cache=True)
def _cached_responses(table, source_x, source_y, pixel_x, pixel_y,
                      map_nx, map_ny, extent, steps):
    n_source, n_pixel = pixel_x.shape
    out = np.empty((n_source,2,n_pixel),dtype=np.float64)
    spatial_ny, spatial_nx = table.shape[:2]
    last = table.shape[-1]-1
    for row in range(n_source):
        sx, sy = source_x[row], source_y[row]
        if not (-.5 <= sx <= map_nx-.5 and -.5 <= sy <= map_ny-.5):
            raise ValueError('Source lies outside calibrated response-cache map')
        gx = (sx+.5)/map_nx*(spatial_nx-1)
        gy = (sy+.5)/map_ny*(spatial_ny-1)
        ix,iy = min(int(gx),spatial_nx-2),min(int(gy),spatial_ny-2)
        fx,fy = gx-ix,gy-iy
        for pixel in range(n_pixel):
            dx = (pixel_x[row,pixel]-sx+extent)*steps
            dy = (pixel_y[row,pixel]-sy+extent)*steps
            if not (0 <= dx <= last and 0 <= dy <= last):
                raise ValueError('Cutout exceeds response-cache displacement support')
            jx,jy = min(int(dx),last-1),min(int(dy),last-1)
            ux,uy = dx-jx,dy-jy
            for member in range(2):
                value = 0.0
                for cy in range(2):
                    wy = fy if cy else 1-fy
                    for cx in range(2):
                        wx = fx if cx else 1-fx
                        im = table[iy+cy,ix+cx,member]
                        local = ((1-ux)*(1-uy)*im[jy,jx] + ux*(1-uy)*im[jy,jx+1]
                                 + ux*uy*im[jy+1,jx+1] + (1-ux)*uy*im[jy+1,jx])
                        value += wx*wy*local
                out[row,member,pixel] = value
    return out


@dataclass(frozen=True)
class CachedPairedPSF:
    """Map-specific interpolation of separately integrated published components.

    The table contains exact PairedPSF quadrature on a deterministic grid of
    source positions and pixel offsets. Interpolation never rescales template
    sums. Validate template and fitted-flux/noise errors for each map before
    production use. Memory is explicit in table.nbytes; construction refuses
    tables above 1 GiB. No cache file or scientific product is written.
    """
    table: np.ndarray
    shape: tuple
    extent: float
    subpixel_steps: int
    integration_order: int

    @classmethod
    def from_map(cls,wcs,shape,components,*,half_size=15,integration_order=8,
                 spatial_nodes=5,subpixel_steps=16):
        for value in (half_size,integration_order,spatial_nodes,subpixel_steps):
            if not isinstance(value,int) or value < 1:
                raise ValueError('Cache geometry must use positive integer dimensions')
        if spatial_nodes < 2 or len(shape)!=2 or min(shape)<=0:
            raise ValueError('Cache requires at least two spatial nodes and a positive map shape')
        extent=half_size+.5
        count=int(2*extent*subpixel_steps)+1
        nbytes=spatial_nodes**2*2*count**2*8
        if nbytes > 1024**3:
            raise ValueError('Response table exceeds 1 GiB budget')
        offsets=np.linspace(-extent,extent,count)
        yy,xx=np.meshgrid(offsets,offsets,indexing='ij')
        table=np.empty((spatial_nodes,spatial_nodes,2,count,count),dtype=np.float64)
        psf=PairedPSF(tuple(components))
        for iy,sy in enumerate(np.linspace(-.5,shape[0]-.5,spatial_nodes)):
            for ix,sx in enumerate(np.linspace(-.5,shape[1]-.5,spatial_nodes)):
                source=wcs.pixel_to_world(np.array([sx]),np.array([sy]))[:,None,None]
                table[iy,ix]=psf(wcs,source,(xx+sx)[None],(yy+sy)[None],integration_order)[0]
        table.flags.writeable=False
        return cls(table,tuple(shape),extent,subpixel_steps,integration_order)

    def evaluate_xy(self,source_x,source_y,pixel_x,pixel_y):
        sx,sy=np.asarray(source_x,dtype=float),np.asarray(source_y,dtype=float)
        x,y=np.broadcast_arrays(np.asarray(pixel_x,dtype=float),np.asarray(pixel_y,dtype=float))
        if (sx.ndim!=1 or sy.shape!=sx.shape or x.ndim!=3 or x.shape[0]!=sx.size
                or not all(np.all(np.isfinite(a)) for a in (sx,sy,x,y))):
            raise ValueError('Expected finite source vectors and (n,ny,nx) pixel coordinates')
        if not sx.size:
            return np.empty((0,2,*x.shape[1:]),dtype=np.float64)
        out=_cached_responses(self.table,sx,sy,x.reshape(sx.size,-1),y.reshape(sx.size,-1),
                              self.shape[1],self.shape[0],self.extent,self.subpixel_steps)
        return out.reshape((sx.size,2,*x.shape[1:]))

    def __call__(self,wcs,source,pixel_x,pixel_y,integration_order):
        if integration_order!=self.integration_order:
            raise ValueError('Requested quadrature differs from the response-cache definition')
        sx,sy=wcs.world_to_pixel(source)
        return self.evaluate_xy(np.asarray(sx).reshape(-1),np.asarray(sy).reshape(-1),pixel_x,pixel_y)


def published_components_for_map(config,record):
    """Select the two published P70 components from retained scan metadata.

    The map supplies one explicit SCANDIR/POSANGLE pair; original observation
    contexts supply the other. No orientation is inferred from a filename or
    filled from pilot numbers. Ambiguous or missing directions fail explicitly.
    """
    from .maps import raw_root
    header=fits.Header.fromstring(record.wcs_header)
    if (record.band!='P70' or int(record.mode)!=31 or record.scan_speed!=60
            or not np.isclose(float(header.get('PIXFRAC',np.nan)),.1)):
        raise ValueError('Published factory supports only P70 parallel60 pixfrac=.1 maps')
    orientations={}
    def add(h,source):
        direction=str(h.get('SCANDIR','')).strip().lower()
        angle=float(h.get('POSANGLE',np.nan))
        if direction not in ('nominal','orthogonal') or not np.isfinite(angle):
            raise ValueError(f'Missing scan direction/position angle in {source}')
        if direction in orientations and abs(orientations[direction]-angle)>1e-8:
            raise ValueError(f'Ambiguous {direction} scan orientation in retained metadata')
        orientations[direction]=angle
    add(header,'map header')
    root=raw_root(config)
    for obsid in record.obsids:
        for path in sorted((root/str(obsid)).glob(f'hpacs{obsid}obs*.fits.gz')):
            with fits.open(path,mode='readonly') as hdus:
                h=hdus[0].header
                if str(h.get('OBS_ID',h.get('OBSID','')))!=str(obsid):
                    raise ValueError(f'Observation context ID differs from indexed scan: {path}')
                add(h,path)
    if set(orientations)!= {'nominal','orthogonal'}:
        raise ValueError('Both original nominal/orthogonal orientations must be retained')
    return tuple(PublishedPSF.read_p70_parallel60(
        root/'calibration'/'psf'/f'psf20_blu_60para_vesta_od345_ama{angle}_0p10.fits.gz',
        position_angle_deg=orientations[direction])
        for direction,angle in (('nominal','+42'),('orthogonal','-42')))


def calibration_factory(config,record,*,half_size=12,integration_order=8,
                        spatial_nodes=5,subpixel_steps=16):
    """Return (map response cache, exact paired PSF) from retained metadata."""
    from astropy.wcs import WCS
    components=published_components_for_map(config,record)
    exact=PairedPSF(components)
    cached=CachedPairedPSF.from_map(WCS(fits.Header.fromstring(record.wcs_header)).celestial,
                                  record.shape,components,half_size=half_size,
                                  integration_order=integration_order,
                                  spatial_nodes=spatial_nodes,subpixel_steps=subpixel_steps)
    return cached,exact
