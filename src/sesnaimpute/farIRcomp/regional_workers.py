"""Streaming within-map calibration pieces using the package's existing pool.

Bind only small metadata and bare numpy arrays into the worker partial; joblib
can then mmap the immutable response table instead of pickling a PSF object.
Each process opens one map and rebuilds one detected-neighbour tree per pass.
"""
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
import time

import numpy as np

from sesnaimpute import pieces
from .maps import open_map
from .geometry import MapGrid, SesnaNeighbours, sample_cells
from .psf import CachedPairedPSF
from .completeness import calibrate_cell_scatter

_STATE = {}


@dataclass(frozen=True)
class CalibrationPiece:
    cell_ids: np.ndarray
    mean_mjy: np.ndarray
    scatter_mjy: np.ndarray
    pixel_sigma_rms_mjy: np.ndarray
    n_positions: np.ndarray
    covered_fraction: np.ndarray
    excluded_fraction: np.ndarray
    draws_used: np.ndarray


def clear_worker_state():
    """Close this process's map; useful after serial execution or a map switch."""
    stack = _STATE.get('stack')
    if stack is not None:
        stack.close()
    _STATE.clear()


def _worker_state(config,record,response_table,map_shape,extent,subpixel_steps,
                  integration_order,detected_ra_deg,detected_dec_deg,run_key,pass_key):
    key = (run_key, str(config.data_root), record.map_id, record.path,
           tuple(map_shape), extent, subpixel_steps, integration_order)
    if _STATE.get('key') != key:
        clear_worker_state()
        stack = ExitStack()
        try:
            image = stack.enter_context(open_map(config,record))
            response_table.flags.writeable = False
            psf = CachedPairedPSF(response_table,tuple(map_shape),extent,
                                 subpixel_steps,integration_order)
            grid = MapGrid.from_wcs(image.wcs,image.image.shape)
        except BaseException:
            stack.close()
            raise
        _STATE.update(key=key,stack=stack,image=image,psf=psf,grid=grid)
    if _STATE.get('pass_key') != pass_key:
        _STATE['neighbours'] = SesnaNeighbours(detected_ra_deg,detected_dec_deg)
        _STATE['pass_key'] = pass_key
    image,psf,grid,neighbours = (_STATE[k] for k in ('image','psf','grid','neighbours'))
    return image,psf,grid,neighbours


def _calibrate_piece(bounds, *, config, record, response_table, map_shape,
                     extent, subpixel_steps, integration_order, cell_ids,
                     detected_ra_deg, detected_dec_deg, run_key, pass_key,
                     seed, area_draws, max_draws, half_size, annulus_arcsec,
                     min_coverage, beam_arcsec, n_positions, batch_size,
                     deadline):
    from .stage import usable_xy, measure_xy

    image,psf,grid,neighbours = _worker_state(config,record,response_table,map_shape,extent,
        subpixel_steps,integration_order,detected_ra_deg,detected_dec_deg,run_key,pass_key)
    start,stop = bounds
    selected_ids = np.asarray(cell_ids[start:stop],dtype=np.int64).copy()
    n = selected_ids.size
    mean,scatter,rms,covered,excluded = (np.empty(n,dtype=float) for _ in range(5))
    count,draws = (np.empty(n,dtype=np.int64) for _ in range(2))
    def usable(x,y):
        return usable_xy(image,x,y,half_size=half_size,min_coverage=min_coverage)
    for i,cell in enumerate(selected_ids):
        if deadline is not None and time.monotonic() > deadline:
            raise RuntimeError('Regional calibration time budget exhausted within worker')
        selected = sample_cells(grid,np.asarray([cell]),neighbours,beam_arcsec,usable,
                                seed=seed,area_draws=area_draws,max_draws=max_draws,
                                n_positions=n_positions)
        try:
            flux,pixel,_ = measure_xy(image,selected.x.ravel(),selected.y.ravel(),psf,
                                      half_size=half_size,annulus_arcsec=annulus_arcsec,
                                      min_coverage=min_coverage,batch_size=batch_size)
        except RuntimeError as exc:
            raise RuntimeError(f'PACS calibration cell_id={int(cell)}: {exc}') from exc
        stats = calibrate_cell_scatter(flux[None],pixel[None],
                                      np.ones((1,flux.size),dtype=bool))
        mean[i],scatter[i],rms[i],count[i] = (stats.mean_mjy[0],stats.scatter_mjy[0],
                                           stats.pixel_sigma_rms_mjy[0],stats.n_positions[0])
        covered[i] = selected.area.covered_fraction[0]
        excluded[i] = selected.area.excluded_fraction[0]
        draws[i] = selected.draws_used[0]
    return CalibrationPiece(selected_ids,mean,scatter,rms,count,covered,excluded,draws)


def iterate_cell_calibrations(config,record,psf,cell_ids,detected_ra_deg,detected_dec_deg,*,
                              n_jobs,run_key,pass_key,seed,area_draws,max_draws,
                              half_size,annulus_arcsec,min_coverage,beam_arcsec,
                              n_positions=200,batch_size=200,piece_cells=8,
                              deadline=None,progress_stage=None):
    """Stream bounded statistics for only the supplied (changed) cells.

    ``run_key`` uniquely identifies the current parent run and immutable PSF
    definition. ``pass_key`` changes whenever detected coordinates change.
    These are in-memory cache keys, not persisted provenance. Caller must not
    reuse either key for different associated inputs. The parent owns the one
    cache construction and reduces each yielded piece immediately. No new pool
    or all-region result list is created here. Deadline, if supplied, is an
    absolute time.monotonic value shared by local processes.
    """
    if not isinstance(psf,CachedPairedPSF):
        raise TypeError('Parallel calibration requires the parent-built CachedPairedPSF')
    if any(int(v)!=v or v<1 for v in (n_jobs,piece_cells,batch_size,n_positions)):
        raise ValueError('Worker, piece, batch and position counts must be positive integers')
    hash(run_key);hash(pass_key)
    if pass_key is None:
        raise ValueError('pass_key must identify the current detected-source pass')
    cells = np.asarray(cell_ids)
    if cells.ndim!=1 or cells.dtype.kind not in 'iu' or np.unique(cells).size!=cells.size:
        raise ValueError('Changed cell IDs must be a unique integer vector')
    ra,dec = np.asarray(detected_ra_deg,float),np.asarray(detected_dec_deg,float)
    if ra.ndim!=1 or ra.shape!=dec.shape:
        raise ValueError('Detected coordinates must be matching one-dimensional arrays')
    if not cells.size:
        return
    function = partial(_calibrate_piece,config=config,record=record,
        response_table=psf.table,map_shape=psf.shape,extent=psf.extent,
        subpixel_steps=psf.subpixel_steps,integration_order=psf.integration_order,
        cell_ids=cells,detected_ra_deg=ra,detected_dec_deg=dec,
        run_key=run_key,pass_key=pass_key,seed=seed,area_draws=area_draws,max_draws=max_draws,
        half_size=half_size,annulus_arcsec=annulus_arcsec,min_coverage=min_coverage,
        beam_arcsec=beam_arcsec,n_positions=int(n_positions),batch_size=int(batch_size),
        deadline=deadline)
    chunks = ((start,min(start+int(piece_cells),cells.size))
              for start in range(0,cells.size,int(piece_cells)))
    yield from pieces.iterate(function,chunks,int(n_jobs),stage=progress_stage,unit='cell pieces')


@dataclass(frozen=True)
class RecoveryPiece:
    cell_ids: np.ndarray
    flux_mjy: np.ndarray
    pixel_sigma_mjy: np.ndarray
    gain: np.ndarray
    n_candidate: np.ndarray
    n_covered: np.ndarray
    n_excluded: np.ndarray
    draws_used: np.ndarray


def _recover_piece(bounds, *, config,record,response_table,map_shape,extent,
                   subpixel_steps,integration_order,cell_ids,detected_ra_deg,
                   detected_dec_deg,run_key,pass_key,seed,area_draws,max_draws,
                   half_size,annulus_arcsec,min_coverage,beam_arcsec,n_positions,
                   batch_size,deadline):
    from .stage import usable_xy,measure_xy
    image,psf,grid,neighbours = _worker_state(config,record,response_table,map_shape,extent,
        subpixel_steps,integration_order,detected_ra_deg,detected_dec_deg,run_key,pass_key)
    start,stop=bounds
    selected_ids=np.asarray(cell_ids[start:stop],dtype=np.int64).copy()
    n=selected_ids.size
    flux,pixel,gain=(np.empty((n,n_positions),dtype=float) for _ in range(3))
    candidate,covered,excluded,draws=(np.empty(n,dtype=np.int64) for _ in range(4))
    def usable(x,y):
        return usable_xy(image,x,y,half_size=half_size,min_coverage=min_coverage)
    for i,cell in enumerate(selected_ids):
        if deadline is not None and time.monotonic()>deadline:
            raise RuntimeError('Regional recovery time budget exhausted within worker')
        selected=sample_cells(grid,np.asarray([cell]),neighbours,beam_arcsec,usable,
                              seed=seed,area_draws=area_draws,max_draws=max_draws,
                              n_positions=n_positions)
        f,p,_,g=measure_xy(image,selected.x.ravel(),selected.y.ravel(),psf,
                           half_size=half_size,annulus_arcsec=annulus_arcsec,
                           min_coverage=min_coverage,batch_size=batch_size,return_gain=True)
        flux[i],pixel[i],gain[i]=f,p,g
        candidate[i]=selected.area.n_candidate[0]
        covered[i]=selected.area.n_covered[0]
        excluded[i]=selected.area.n_excluded[0]
        draws[i]=selected.draws_used[0]
    return RecoveryPiece(selected_ids,flux,pixel,gain,candidate,covered,excluded,draws)


def iterate_cell_recoveries(config,record,psf,cell_ids,detected_ra_deg,detected_dec_deg,*,
                            n_jobs,run_key,pass_key,seed,area_draws,max_draws,
                            half_size,annulus_arcsec,min_coverage,beam_arcsec,
                            n_positions=200,batch_size=200,piece_cells=8,
                            deadline=None,progress_stage=None):
    """Stream held-out baseline flux/noise/gain trials with fixed-pool area counts.

    Call with a curve seed distinct from calibration and a frozen detected set.
    The parent supplies frozen cell scatter and constructs the empirical CDF;
    workers do not tune endpoints or recompute calibration from recovery trials.
    Cache and deadline contracts match iterate_cell_calibrations.
    """
    if not isinstance(psf,CachedPairedPSF):
        raise TypeError('Parallel recovery requires the parent-built CachedPairedPSF')
    if any(int(v)!=v or v<1 for v in (n_jobs,piece_cells,batch_size,n_positions)):
        raise ValueError('Worker, piece, batch and position counts must be positive integers')
    hash(run_key);hash(pass_key)
    if pass_key is None:
        raise ValueError('pass_key must identify the frozen detected-source pass')
    cells=np.asarray(cell_ids)
    if cells.ndim!=1 or cells.dtype.kind not in 'iu' or np.unique(cells).size!=cells.size:
        raise ValueError('Recovery cell IDs must be a unique integer vector')
    ra,dec=np.asarray(detected_ra_deg,float),np.asarray(detected_dec_deg,float)
    if ra.ndim!=1 or ra.shape!=dec.shape:
        raise ValueError('Detected coordinates must be matching one-dimensional arrays')
    if not cells.size:
        return
    function=partial(_recover_piece,config=config,record=record,response_table=psf.table,
        map_shape=psf.shape,extent=psf.extent,subpixel_steps=psf.subpixel_steps,
        integration_order=psf.integration_order,cell_ids=cells,detected_ra_deg=ra,
        detected_dec_deg=dec,run_key=run_key,pass_key=pass_key,seed=seed,
        area_draws=area_draws,max_draws=max_draws,half_size=half_size,
        annulus_arcsec=annulus_arcsec,min_coverage=min_coverage,beam_arcsec=beam_arcsec,
        n_positions=int(n_positions),batch_size=int(batch_size),deadline=deadline)
    chunks=((start,min(start+int(piece_cells),cells.size))
            for start in range(0,cells.size,int(piece_cells)))
    yield from pieces.iterate(function,chunks,int(n_jobs),stage=progress_stage,unit='recovery pieces')
