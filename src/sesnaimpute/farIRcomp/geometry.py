"""Fixed PACS map cells and equal-area admitted-pixel samples.

Cells have 30-arcsec sides in the map projection, aligned with native pixel
axes. FITS zero-origin pixel centres are integers and image boundaries are
-0.5 and n-0.5. Edge cells are clipped to those boundaries. Coverage holes and
overlap ownership remain explicit caller masks, not properties inferred from
the rectangular image extent.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales, wcs_to_celestial_frame
from scipy.spatial import cKDTree
from scipy.special import erfc  # import before healpy for native-library initialization order
import healpy as hp

from .constants import CELL_ARCSEC


@dataclass(frozen=True)
class MapGrid:
    """One tile's deterministic y-major grid, with its native celestial frame."""

    wcs: object
    shape: tuple
    step_x: float
    step_y: float
    nx: int
    ny: int

    @classmethod
    def from_wcs(cls, wcs, shape):
        if len(shape) != 2 or any(int(v) != v or v <= 0 for v in shape):
            raise ValueError("map shape must contain two positive integers (ny,nx)")
        wcs = wcs.celestial
        if wcs.pixel_n_dim != 2 or wcs.world_axis_physical_types != ["pos.eq.ra", "pos.eq.dec"]:
            raise ValueError("map grid requires a two-dimensional RA/Dec WCS")
        # Validate the native frame, including FK5/J2000 archive maps. Never
        # relabel native RA/Dec as ICRS; centres_icrs transforms explicitly.
        wcs_to_celestial_frame(wcs)
        scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=np.float64) * 3600.0
        if not np.all(np.isfinite(scales) & (scales > 0)):
            raise ValueError("invalid map pixel scale")
        matrix = wcs.pixel_scale_matrix
        cosine = np.dot(matrix[:, 0], matrix[:, 1]) / np.prod(scales / 3600.0)
        if abs(cosine) > 1e-6 or wcs.has_distortion:
            raise ValueError("skewed or distortion-corrected WCS requires explicit cell geometry")
        sx, sy = CELL_ARCSEC / scales
        shape = tuple(int(v) for v in shape)
        return cls(wcs, shape, float(sx), float(sy),
                   int(np.ceil(shape[1] / sx)), int(np.ceil(shape[0] / sy)))

    @property
    def n_cells(self):
        return self.nx * self.ny

    def cell_ids(self, x, y):
        """Native zero-origin pixel coordinates to cell IDs; -1 outside tile."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        valid = np.isfinite(x) & np.isfinite(y)
        valid &= (x >= -0.5) & (x < self.shape[1] - 0.5)
        valid &= (y >= -0.5) & (y < self.shape[0] - 0.5)
        result = np.full(x.shape, -1, dtype=np.int64)
        ix = np.floor((x[valid] + 0.5) / self.step_x).astype(np.int64)
        iy = np.floor((y[valid] + 0.5) / self.step_y).astype(np.int64)
        result[valid] = iy * self.nx + ix
        return result

    def bounds(self, cell_ids):
        """Return x0,x1,y0,y1 for native cells, clipped at tile edges."""
        ids = np.asarray(cell_ids)
        if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= self.n_cells):
            raise ValueError("cell IDs must be valid integers on this map")
        x0 = (ids % self.nx) * self.step_x - 0.5
        y0 = (ids // self.nx) * self.step_y - 0.5
        return x0, np.minimum(x0 + self.step_x, self.shape[1] - 0.5), \
            y0, np.minimum(y0 + self.step_y, self.shape[0] - 0.5)

    def centres_icrs(self, cell_ids):
        """Return ICRS RA/Dec degrees at each clipped cell's centre."""
        x0, x1, y0, y1 = self.bounds(cell_ids)
        sky = self.wcs.pixel_to_world((x0 + x1) / 2, (y0 + y1) / 2).icrs
        return sky.ra.deg, sky.dec.deg

    def sample_pixels(self, cell_ids, n_positions, seed, *, draw_start=0):
        """Uniform projected-area positions, reproducible across cell batches.

        Samples are NOT filtered for coverage. The caller must evaluate the
        coverage and selected-tile masks explicitly. ``seed`` identifies the
        tile; streams are indexed by native cell and draw, independent of the
        order or partition of ``cell_ids``. This samples projected area, whose
        difference from spherical area is not silently corrected here.
        """
        ids = np.asarray(cell_ids)
        if ids.ndim != 1 or self.n_cells >= 2**32:
            raise ValueError("sampling requires a cell vector and fewer than 2**32 map cells")
        if int(n_positions) != n_positions or not 1 <= n_positions < 2**31:
            raise ValueError("n_positions must be an integer in 1..2**31-1")
        if int(draw_start)!=draw_start or draw_start<0 or draw_start+n_positions>=2**31:
            raise ValueError("draw range must stay within this cell's counter stream")
        n_positions,draw_start=int(n_positions),int(draw_start)
        if int(seed) != seed or not 0 <= seed < 2**64:
            raise ValueError("seed must be a uint64-compatible integer")
        x0, x1, y0, y1 = self.bounds(ids)
        counter = (ids.astype(np.uint64)[:, None] << np.uint64(32))
        draws = np.arange(2*draw_start,2*(draw_start+n_positions),dtype=np.uint64)[None,:]
        key = _mix64(np.array([seed], dtype=np.uint64))[0]
        uniform = _uniform((counter + draws) ^ key)
        x = x0[:, None] + (x1 - x0)[:, None] * uniform[:, 0::2]
        y = y0[:, None] + (y1 - y0)[:, None] * uniform[:, 1::2]
        return x, y


def _mix64(counter):
    """SplitMix64 mixing with intentional uint64 wraparound."""
    value = counter + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return value


@dataclass(frozen=True)
class CellSamples:
    """Accepted trials plus an area estimate from a fixed pre-selection pool."""

    cell_ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    area: object
    draws_used: np.ndarray
    raw_covered_fraction: object


class SamplingBudgetError(ValueError):
    """Strict pilot failure retaining every cell's area/count diagnostics."""

    def __init__(self, cell_ids, n_available, area, raw_covered_fraction):
        self.cell_ids = cell_ids.copy()
        self.n_available = n_available.copy()
        self.area = area
        self.raw_covered_fraction = raw_covered_fraction
        super().__init__("accepted sampling budget exhausted; cell IDs=%s counts=%s" %
                         (cell_ids.tolist(), n_available.tolist()))


def sample_cells(grid, cell_ids, neighbours, beam_arcsec, covered_at_pixels, *,
                 seed, area_draws, max_draws, n_positions=200,
                 raw_covered_at_pixels=None):
    """Strict pilot: refill covered, unexcluded trials with an explicit bound.

    The coverage callback takes matching (cell,draw) native x/y arrays and
    returns geometry-only coverage, including selected-tile ownership if needed.
    Extraction validity must NOT influence this mask. Area uses the first fixed
    ``area_draws`` candidates, irrespective of how soon accepted trials fill.
    Failure to fill any cell raises SamplingBudgetError carrying EVERY cell's
    counts and area diagnostics; production must not catch and drop such cells.
    No partial curve or silent extraction-based rejection is produced. Caller
    bounds batch size (the pilot runner uses one cell at a time).
    Current completeness supplies DetectedPacsNeighbours, shared with grouping;
    an all-catalogue tree is permitted only for the explicit old-rule comparison.
    """
    if (int(area_draws) != area_draws or int(max_draws) != max_draws
            or not 1 <= area_draws <= max_draws
            or int(n_positions) != n_positions or not 1 <= n_positions <= 32767):
        raise ValueError("require positive integer draws and positions, area_draws<=max_draws")
    area_draws, max_draws, n_positions = map(int, (area_draws, max_draws, n_positions))
    ids = np.asarray(cell_ids)
    if ids.ndim != 1 or np.unique(ids).size != ids.size:
        raise ValueError("cell IDs must be a unique vector")
    if max_draws>=2**31:
        raise ValueError("max_draws exceeds deterministic per-cell stream")
    outx=np.full((ids.size,n_positions),np.nan)
    outy=outx.copy();counts=np.zeros(ids.size,np.int64)
    filled=np.zeros(ids.size,np.int64);used=np.zeros(ids.size,np.int64)
    cursor=0;block=max(area_draws,2*n_positions);raw_fraction=None
    while cursor<max_draws:
        active=np.flatnonzero(filled<n_positions)
        if not active.size and cursor:
            break
        size=min(block,max_draws-cursor)
        x,y=grid.sample_pixels(ids[active],size,seed,draw_start=cursor)
        covered=np.asarray(covered_at_pixels(x,y),dtype=bool)
        if covered.shape!=x.shape:
            raise ValueError("coverage callback must preserve sample shape")
        sky=grid.wcs.pixel_to_world(x,y).icrs
        excluded=neighbours.excluded(sky.ra.deg,sky.dec.deg,beam_arcsec)
        if cursor==0:
            area=exclusion_area(covered[:,:area_draws],excluded[:,:area_draws])
            if raw_covered_at_pixels is not None:
                raw=np.asarray(raw_covered_at_pixels(x[:,:area_draws],y[:,:area_draws]),dtype=bool)
                if raw.shape!=(ids.size,area_draws):
                    raise ValueError("raw coverage callback must preserve sample shape")
                if np.any(covered[:,:area_draws]&~raw):
                    raise ValueError("usable coverage cannot exceed raw coverage")
                raw_fraction=raw.mean(axis=1)
        accepted=covered&~excluded
        counts[active]+=accepted.sum(axis=1)
        for local,row in enumerate(active):
            selected=np.flatnonzero(accepted[local])[:n_positions-filled[row]]
            stop=filled[row]+selected.size
            outx[row,filled[row]:stop]=x[local,selected]
            outy[row,filled[row]:stop]=y[local,selected]
            filled[row]=stop
            if selected.size:
                used[row]=cursor+selected[-1]+1
        cursor+=size
        block=min(max(block*2,1),16384)
    if np.any(counts < n_positions):
        raise SamplingBudgetError(ids, counts, area, raw_fraction)
    return CellSamples(ids.copy(),outx,outy,area,used,raw_fraction)


def _uniform(counter):
    """Uniform float64 values from the upper 53 mixed counter bits."""
    return (_mix64(counter) >> np.uint64(11)).astype(np.float64) * (1.0 / 2**53)


def certify_cell_empty(grid,cell_id,neighbours,covered_at_pixels,beam_arcsec,*,max_boxes=100000):
    """Conservatively prove no usable, unexcluded area in a TAN-map cell.

    Usable-centre coverage is constant in each nearest-native-pixel rectangle.
    A spherical cap smaller than a hemisphere is geodesically convex, and TAN
    rectangle edges are great-circle arcs. Thus all four corners strictly
    inside ONE detected-source cap prove the entire rectangle excluded. For
    overlapping caps, subdivide rectangles until this proof succeeds. Any
    eligible centre or unresolved numerical/budget boundary returns False,
    meaning NOT PROVEN EMPTY, never a fabricated no-curve classification.
    """
    if list(grid.wcs.wcs.ctype)!=['RA---TAN','DEC--TAN'] or not 0<beam_arcsec<90*3600:
        return False
    x0,x1,y0,y1=(float(v[0]) for v in grid.bounds(np.asarray([cell_id],dtype=np.int64)))
    ix=np.arange(int(np.floor(x0+.5)),int(np.ceil(x1+.5)))
    iy=np.arange(int(np.floor(y0+.5)),int(np.ceil(y1+.5)))
    xx,yy=np.meshgrid(ix,iy)
    usable=np.asarray(covered_at_pixels(xx,yy),bool)
    if usable.shape!=xx.shape:
        raise ValueError('coverage callback must preserve pixel mesh shape')
    if not np.any(usable):
        return True
    if not neighbours.n_sources:
        return False
    boxes=[(max(x0,x-.5),min(x1,x+.5),max(y0,y-.5),min(y1,y+.5))
           for x,y in zip(xx[usable],yy[usable])]
    limit=2*np.sin(np.deg2rad(beam_arcsec/3600)/2)-64*np.finfo(float).eps
    visited=0
    while boxes:
        visited+=1
        if visited>max_boxes:
            return False
        a,b,c,d=boxes.pop()
        if b<=a or d<=c:
            continue
        sky=grid.wcs.pixel_to_world(np.array([a,b,a,b,(a+b)/2]),
                                    np.array([c,c,d,d,(c+d)/2])).icrs
        vectors,_=_unit_vectors(sky.ra.deg,sky.dec.deg)
        distance,nearest=neighbours._tree.query(vectors[-1],k=1)
        if distance>limit:
            return False
        anchor=neighbours._tree.data[nearest]
        if np.all(np.linalg.norm(vectors[:4]-anchor,axis=1)<limit):
            continue
        if min(b-a,d-c)<1e-9:
            return False
        mx,my=(a+b)/2,(c+d)/2
        boxes.extend([(a,mx,c,my),(mx,b,c,my),(a,mx,my,d),(mx,b,my,d)])
    return True


def row_lookup(n_cells, covered_cell_ids, row_start=0):
    """Return sorted covered native IDs and their global int32 row lookup.

    The caller assigns stable tile order and supplies each tile's row_start.
    Duplicate cell IDs are an error, so accidental overlap concatenation is
    visible. Unselected cells carry -1.
    """
    if int(n_cells) != n_cells or n_cells < 0 or int(row_start) != row_start or row_start < 0:
        raise ValueError("n_cells and row_start must be nonnegative integers")
    n_cells, row_start = int(n_cells), int(row_start)
    ids = np.asarray(covered_cell_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("covered cell IDs must be an integer vector")
    if np.any(ids < 0) or np.any(ids >= n_cells):
        raise ValueError("covered cell IDs are outside the map")
    ids = np.sort(ids.astype(np.int64))
    if np.any(np.diff(ids) == 0):
        raise ValueError("duplicate covered cell IDs")
    if row_start + ids.size > np.iinfo(np.int32).max + 1:
        raise ValueError("global curve rows exceed int32")
    lookup = np.full(n_cells, -1, dtype=np.int32)
    lookup[ids] = np.arange(row_start, row_start + ids.size, dtype=np.int64)
    return ids, lookup


def rows_for_cells(cell_ids, lookup, covered):
    """Map positions on this selected tile to global rows, -1 if uncovered."""
    ids, lookup = np.asarray(cell_ids), np.asarray(lookup)
    covered = np.asarray(covered, dtype=bool)
    if ids.shape != covered.shape or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("integer cell IDs and coverage must have identical shapes")
    if lookup.ndim != 1 or lookup.dtype != np.int32:
        raise ValueError("lookup must be the int32 native-cell row table")
    if np.any(covered & ((ids < 0) | (ids >= lookup.size))):
        raise ValueError("covered position lies outside its selected tile")
    rows = np.full(ids.shape, -1, dtype=np.int32)
    rows[covered] = lookup[ids[covered]]
    if np.any(covered & (rows < 0)):
        raise ValueError("covered position has no tabulated cell")
    return rows


def admitted_sample_points(admitted):
    """ICRS RA/Dec (n_pixel,256), preserving Galactic NESTED nside-512 order."""
    admitted = np.asarray(admitted)
    if admitted.ndim != 1 or not np.issubdtype(admitted.dtype, np.integer):
        raise ValueError("admitted pixels must be an integer vector")
    if np.any(admitted < 0) or np.any(admitted >= hp.nside2npix(512)):
        raise ValueError("invalid nside-512 pixel")
    if np.unique(admitted).size != admitted.size:
        raise ValueError("duplicate admitted pixels")
    children = admitted.astype(np.int64)[:, None] * 256 + np.arange(256)[None, :]
    lon, lat = hp.pix2ang(8192, children, nest=True, lonlat=True)
    icrs = SkyCoord(l=lon, b=lat, unit="deg", frame="galactic").icrs
    return icrs.ra.deg, icrs.dec.deg


def regional_cell_ids(grid,admitted,extra_cell_ids=()):
    """Conservative native-cell boxes around the admitted HEALPix footprint.

    Sample each nside512 boundary with16subdivisions/edge and pad the resulting
    native-cell box by one full cell. These are conservative processing boxes,
    not a new admission definition: final area averages use the exact existing
    256nested-child samples and explicit masks. Covered source/sample cell IDs
    may be supplied to guarantee their inclusion at regional boundaries.
    """
    admitted=np.asarray(admitted,dtype=np.int64)
    if admitted.ndim!=1 or np.any(admitted<0) or np.any(admitted>=hp.nside2npix(512)):
        raise ValueError('Invalid admitted nside512 pixels')
    selected=np.zeros(grid.n_cells,bool)
    if admitted.size:
        vectors=hp.boundaries(512,admitted,step=16,nest=True).transpose(0,2,1)
        lon,lat=hp.vec2ang(vectors.reshape(-1,3),lonlat=True)
        sky=SkyCoord(l=lon,b=lat,unit='deg',frame='galactic').icrs
        x,y=grid.wcs.world_to_pixel(sky)
        x,y=x.reshape(admitted.size,-1),y.reshape(admitted.size,-1)
        if not np.all(np.isfinite(x)&np.isfinite(y)):
            raise ValueError('Admitted footprint crosses unsupported map projection')
        for xx,yy in zip(x,y):
            ax=max(0,int(np.floor((xx.min()+.5)/grid.step_x))-1)
            bx=min(grid.nx,int(np.floor((xx.max()+.5)/grid.step_x))+2)
            ay=max(0,int(np.floor((yy.min()+.5)/grid.step_y))-1)
            by=min(grid.ny,int(np.floor((yy.max()+.5)/grid.step_y))+2)
            if bx>ax and by>ay:
                rows=np.arange(ay,by)[:,None]*grid.nx+np.arange(ax,bx)[None,:]
                selected[rows.ravel()]=True
    extra=np.asarray(extra_cell_ids,dtype=np.int64)
    if np.any(extra<0) or np.any(extra>=grid.n_cells):
        raise ValueError('Extra regional cells outside this map')
    selected[extra]=True
    return np.flatnonzero(selected)


def select_deeper(tile_index, position_indices, local_rms, covered,
                  selected_tile, selected_rms):
    """Update the winning tile at candidate positions, in place.

    Initialize ``selected_tile`` to int32 -1 and ``selected_rms`` to float64
    +inf, one entry per source or flattened HEALPix sample. ``tile_index`` is
    its ordinal in the index's stable map_id order. Unique candidate indices,
    coverage, and local rms have matching vector shapes. The lowest positive
    finite rms wins; equal rms chooses the smaller tile ordinal regardless of
    processing order. Uncovered candidates cannot win. Covered candidates with
    invalid rms cannot win either; their count is returned for explicit reporting.
    This allocates only candidate-sized working arrays, never a position-by-tile
    matrix. All comparisons retain float64 precision until ownership is final.
    """
    indices = _position_indices(position_indices, selected_tile, tile_index)
    if (not isinstance(selected_rms, np.ndarray) or selected_rms.dtype != np.float64
            or selected_rms.shape != selected_tile.shape or not selected_rms.flags.writeable
            or not selected_tile.flags.writeable):
        raise ValueError("selection arrays must be writable matching int32/float64 vectors")
    rms = np.asarray(local_rms, dtype=np.float64)
    coverage = np.asarray(covered, dtype=bool)
    if rms.shape != indices.shape or coverage.shape != indices.shape:
        raise ValueError("candidate indices, local rms and coverage must have matching shapes")
    old_tile = selected_tile[indices]
    old_rms = selected_rms[indices]
    if np.any(old_tile < -1) or np.any((old_tile == -1) & ~np.isposinf(old_rms)):
        raise ValueError("unselected positions require tile=-1 and rms=+inf")
    if np.any((old_tile >= 0) & (~np.isfinite(old_rms) | (old_rms <= 0))):
        raise ValueError("selected positions require positive finite rms")
    valid_rms = np.isfinite(rms) & (rms > 0)
    replace = coverage & valid_rms
    replace &= (rms < old_rms) | ((rms == old_rms) & (tile_index < old_tile))
    chosen = indices[replace]
    selected_tile[chosen] = tile_index
    selected_rms[chosen] = rms[replace]
    return int(np.count_nonzero(coverage & ~valid_rms))


def selected_curve_rows(tile_index, position_indices, cell_ids, lookup, selected_tile):
    """Return (position_indices, curve_rows) for this tile's winning positions.

    Call only after ownership selection is complete. ``cell_ids`` belongs to
    this tile and is aligned with its candidate position indices. A covered
    winner without a tabulated cell is an error; no nearby or overlapping map's
    curve is substituted. The parent scatters these rows into its initially -1
    SOURCE_ROW column or flattened HEALPix sample-row array, one tile at a time.
    """
    indices = _position_indices(position_indices, selected_tile, tile_index)
    cells = np.asarray(cell_ids)
    if cells.shape != indices.shape or not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("candidate cell IDs must match position indices")
    owned = selected_tile[indices] == tile_index
    ids = cells[owned]
    rows = rows_for_cells(ids, lookup, np.ones(ids.shape, dtype=bool))
    return indices[owned], rows


def _position_indices(position_indices, selected_tile, tile_index):
    """Validate the shared source/sample index convention before any update."""
    if (not isinstance(selected_tile, np.ndarray) or selected_tile.ndim != 1
            or selected_tile.dtype != np.int32):
        raise ValueError("selected_tile must be a one-dimensional int32 array")
    if (not isinstance(tile_index, (int, np.integer)) or tile_index < 0
            or tile_index > np.iinfo(np.int32).max):
        raise ValueError("tile index must be a nonnegative int32-compatible integer")
    indices = np.asarray(position_indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("candidate position indices must be an integer vector")
    if np.any(indices < 0) or np.any(indices >= selected_tile.size):
        raise ValueError("candidate position index is out of bounds")
    if np.unique(indices).size != indices.size:
        raise ValueError("duplicate candidate position indices within a tile")
    return indices


def _unit_vectors(ra_deg, dec_deg):
    """ICRS degrees to unit vectors, with no tangent-plane approximation."""
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    if ra.shape != dec.shape or not np.all(np.isfinite(ra)) or not np.all(np.isfinite(dec)):
        raise ValueError('RA/Dec must have identical shapes and finite values')
    if np.any(np.abs(dec) > 90):
        raise ValueError('Declination must lie in [-90,90] degrees')
    longitude = np.deg2rad(np.remainder(ra, 360.0)).ravel()
    latitude = np.deg2rad(dec).ravel()
    cosine = np.cos(latitude)
    vectors = np.empty((longitude.size, 3), dtype=np.float64)
    vectors[:, 0] = cosine * np.cos(longitude)
    vectors[:, 1] = cosine * np.sin(longitude)
    vectors[:, 2] = np.sin(latitude)
    # At the poles longitude is undefined; remove trig roundoff dependence on it.
    polar = np.abs(dec.ravel()) == 90.0
    vectors[polar, :2] = 0.0
    return vectors, ra.shape


class SesnaNeighbours:
    """Spherical catalogue-position tree for explicit historical comparisons.

    Build once per region (or worker lifetime), then query tile-sized batches.
    Production completeness must use DetectedPacsNeighbours instead. This
    all-SESNA tree is retained only to quantify the superseded exclusion rule.
    This is a positional API, deliberately not a blend finder:
    a sample exactly at a SESNA source is excluded, including a self-match.
    Real-source grouping must use a separate neighbour query that removes its
    own catalogue row. No FIR positions or fluxes enter this tree.
    """

    def __init__(self, ra_deg, dec_deg):
        vectors, shape = _unit_vectors(ra_deg, dec_deg)
        if len(shape) != 1:
            raise ValueError('SESNA catalogue positions must be one-dimensional vectors')
        vectors.flags.writeable = False
        self._tree = cKDTree(vectors, copy_data=False)

    @property
    def n_sources(self):
        return self._tree.n

    def excluded(self, ra_deg, dec_deg, beam_arcsec):
        """Boolean mask for samples at angular separation <= the supplied beam.

        Inputs may be arbitrary matching shapes; all queries run in one batched
        tree call with one thread. Caller controls batch size to bound temporary
        vectors (24 bytes per sample plus query outputs). The chord threshold is
        2*sin(beam/2); no projected RA/Dec approximation is used. A 16-epsilon
        absolute chord allowance includes mathematically exact boundaries despite
        float64 trig/subtraction roundoff (below 1e-9 arcsec at PACS beam sizes).
        Coverage is intentionally separate from this geometric mask.
        """
        radius = np.asarray(beam_arcsec)
        if radius.ndim != 0 or not np.isfinite(radius) or not 0 < radius <= 180 * 3600:
            raise ValueError('beam_arcsec must be a positive finite scalar <=180 degrees')
        vectors, shape = _unit_vectors(ra_deg, dec_deg)
        if not self.n_sources or not vectors.shape[0]:
            return np.zeros(shape, dtype=bool)
        chord = 2.0 * np.sin(np.deg2rad(float(radius) / 3600.0) / 2.0)
        limit = chord + 16.0 * np.finfo(np.float64).eps
        distance, _ = self._tree.query(vectors, k=1, workers=1,
                                      distance_upper_bound=np.nextafter(limit, np.inf))
        return (distance <= limit).reshape(shape)


@dataclass(frozen=True)
class SourceCandidates:
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    source_row: np.ndarray
    catalogue: np.ndarray


class DetectedPacsNeighbours(SesnaNeighbours):
    """Frozen first-pass PACS detections, shared by grouping and exclusion.

    Keep this same detected index set after grouped refits. Undetected SESNA
    candidates do not enter the tree; a later inseparable SOURCE_ROW=-1 does
    not retrospectively remove a first-pass detection from this set.
    """

    def __init__(self, candidates, detected):
        mask = np.asarray(detected, dtype=bool)
        if not isinstance(candidates, SourceCandidates) or mask.shape != candidates.ra_deg.shape:
            raise ValueError("detection mask must match explicit source candidates")
        super().__init__(candidates.ra_deg[mask], candidates.dec_deg[mask])
        self.indices = np.flatnonzero(mask)
        self.source_row = candidates.source_row[mask].copy()
        self.catalogue = candidates.catalogue[mask].copy()
        for value in (self.indices,self.source_row,self.catalogue):
            value.flags.writeable = False


def load_sesna_candidates(config, centre_ra_deg, centre_dec_deg, radius_deg, *,
                          chunk_rows=100000):
    """Read all regional source catalogues, retaining a conservative sky cap.

    Caller must enclose EVERY candidate position plus its beam margin in the
    supplied cap. All source classes/fluxes are retained, including overlapping
    catalogue rows. Scan all existing regional products, not just the map's
    named region, so neighbours across catalogue boundaries are included.
    """
    if not np.isfinite(radius_deg) or not 0 < radius_deg <= 180:
        raise ValueError("cap radius must be in (0,180] degrees")
    if int(chunk_rows) != chunk_rows or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    chunk_rows = int(chunk_rows)
    centre, _ = _unit_vectors(np.asarray([centre_ra_deg]), np.asarray([centre_dec_deg]))
    paths = sorted((Path(config.data_root) / "catalog" / "sesna").glob(
        "sources_sesna_source__*.hdf5"))
    if not paths:
        raise FileNotFoundError("no curated SESNA source catalogues under data_root/catalog/sesna")
    chord = 2 * np.sin(np.deg2rad(radius_deg) / 2) + 32 * np.finfo(float).eps
    ras, decs, rows, catalogues = [], [], [], []
    n_read = 0
    for path in paths:
        with h5py.File(path, "r") as handle:
            ra, dec = handle["RA_DEG"], handle["DEC_DEG"]
            if ra.ndim != 1 or ra.shape != dec.shape:
                raise ValueError("invalid SESNA coordinate datasets: %s" % path)
            n_read += ra.size
            for start in range(0, ra.size, int(chunk_rows)):
                r, d = ra[start:start + chunk_rows], dec[start:start + chunk_rows]
                vectors, _ = _unit_vectors(r, d)
                keep = np.linalg.norm(vectors - centre, axis=1) <= chord
                ras.append(r[keep])
                decs.append(d[keep])
                rows.append(np.arange(start,start+r.size,dtype=np.int64)[keep])
                catalogues.append(np.full(keep.sum(),path.name,dtype=object))
    values = (np.concatenate(ras) if ras else np.empty(0),
              np.concatenate(decs) if decs else np.empty(0),
              np.concatenate(rows) if rows else np.empty(0,dtype=np.int64),
              np.concatenate(catalogues) if catalogues else np.empty(0,dtype=object))
    for value in values:
        value.flags.writeable = False
    candidates = SourceCandidates(*values)
    return candidates, {"catalogues": len(paths), "rows_read": n_read,
                        "rows_retained": candidates.ra_deg.size}


def load_sesna_neighbours(config, centre_ra_deg, centre_dec_deg, radius_deg, *,
                          chunk_rows=100000):
    """Historical all-SESNA exclusion comparison; not current completeness."""
    candidates, audit = load_sesna_candidates(config,centre_ra_deg,centre_dec_deg,
                                             radius_deg,chunk_rows=chunk_rows)
    return SesnaNeighbours(candidates.ra_deg,candidates.dec_deg), audit


@dataclass(frozen=True)
class ExclusionArea:
    """Uniform-pool area counts, separate from accepted recovery-trial counts."""

    n_candidate: np.ndarray
    n_covered: np.ndarray
    n_excluded: np.ndarray
    n_available: np.ndarray
    covered_fraction: np.ndarray
    excluded_fraction: np.ndarray


def exclusion_area(covered, pacs_excluded):
    """Measure excluded covered area from a fixed uniform pre-exclusion pool.

    Both masks are (n_cell,n_candidate). Use every candidate in this independent
    uniform pool, never an adaptively refilled/truncated set of 200 accepted
    trials. n_excluded counts covered AND excluded samples; n_available counts
    covered AND unexcluded. excluded_fraction is n_excluded/n_covered (NaN if
    no covered candidates). covered_fraction is n_covered/n_candidate (NaN
    for an empty candidate pool). All counts are int64; fractions float64.
    A fully excluded covered cell has excluded_fraction=1 and n_available=0;
    it supplies no conditional completeness curve, not a zero-valued curve.
    """
    coverage = np.asarray(covered, dtype=bool)
    excluded = np.asarray(pacs_excluded, dtype=bool)
    if coverage.ndim != 2 or excluded.shape != coverage.shape:
        raise ValueError('coverage and exclusion masks must match (n_cell,n_candidate)')
    n_candidate = np.full(coverage.shape[0], coverage.shape[1], dtype=np.int64)
    n_covered = np.count_nonzero(coverage, axis=1).astype(np.int64)
    n_excluded = np.count_nonzero(coverage & excluded, axis=1).astype(np.int64)
    covered_fraction = np.full(n_covered.shape, np.nan, dtype=np.float64)
    excluded_fraction = np.full(n_covered.shape, np.nan, dtype=np.float64)
    np.divide(n_covered, n_candidate, out=covered_fraction, where=n_candidate > 0)
    np.divide(n_excluded, n_covered, out=excluded_fraction, where=n_covered > 0)
    return ExclusionArea(n_candidate, n_covered, n_excluded,
                         n_covered - n_excluded, covered_fraction, excluded_fraction)
