"""Spherical physical-source groups anchored only on detected PACS sources."""
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from .geometry import _unit_vectors


@dataclass(frozen=True)
class DetectedGroups:
    """Input-row identity and deterministic nonsingleton connected components.

    ``group_index`` is -1 for singletons; ``members`` contains input-row arrays
    for nonsingleton groups, ordered by their smallest input row. ``n_blend``
    counts physical catalogue sources, including the target, never PSF columns.
    ``canonical_source_ids`` retains the supplied unique IDs in input order.
    """
    canonical_source_ids: np.ndarray
    group_index: np.ndarray
    members: tuple
    n_blend: np.ndarray


@njit(cache=True)
def _union_edges(parent, left, right):
    for k in range(left.size):
        a, b = left[k], right[k]
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a != b:
            parent[max(a,b)] = min(a,b)


@njit(cache=True)
def _roots(parent):
    for k in range(parent.size):
        a = k
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        parent[k] = a
    return parent


def detected_groups(ra_deg, dec_deg, firstpass_detected, source_ids, *,
                    beam_arcsec, query_batch=4096):
    """Connect neighbours iff at least one endpoint is first-pass detected.

    Distances are spherical and inclusive at the calibrated beam boundary.
    Query only detected anchors against the full catalogue tree, in bounded
    batches. Distinct IDs at coincident coordinates remain distinct sources;
    repeated catalogue IDs are rejected explicitly instead of self-matching.
    Faint sources can bridge two detected anchors into one component, but no
    edge is ever introduced between two undetected sources. With no detections
    all rows are singletons, so members is empty and group_index is all -1.
    Caller must include neighbouring sources across regional boundaries.
    """
    vectors, shape = _unit_vectors(ra_deg, dec_deg)
    if len(shape) != 1:
        raise ValueError('source coordinates must be one-dimensional')
    ids = np.asarray(source_ids)
    detected = np.asarray(firstpass_detected)
    if ids.shape != shape or detected.shape != shape or detected.dtype != np.bool_:
        raise ValueError('IDs and boolean first-pass detections must match source coordinates')
    if ids.dtype.kind not in 'iuUS':
        raise ValueError('canonical source IDs must be integer or string values')
    if np.unique(ids).size != ids.size:
        raise ValueError('duplicate canonical source IDs; deduplicate catalogue identities first')
    radius = np.asarray(beam_arcsec)
    if radius.ndim or not np.isfinite(radius) or not 0 < radius <= 180*3600:
        raise ValueError('beam_arcsec must be a positive finite scalar <=180 degrees')
    if int(query_batch) != query_batch or query_batch < 1:
        raise ValueError('query_batch must be a positive integer')
    query_batch = int(query_batch)
    n = vectors.shape[0]
    parent = np.arange(n,dtype=np.int64)
    anchors = np.flatnonzero(detected)
    if anchors.size:
        tree = cKDTree(vectors)
        limit = 2*np.sin(np.deg2rad(float(radius)/3600)/2) + 16*np.finfo(float).eps
        for start in range(0,anchors.size,query_batch):
            rows = anchors[start:start+query_batch]
            neighbours = tree.query_ball_point(vectors[rows], np.nextafter(limit,np.inf),
                                               workers=1,return_sorted=True)
            lengths = np.fromiter((len(v) for v in neighbours),dtype=np.int64,count=rows.size)
            right = np.concatenate(neighbours).astype(np.int64,copy=False)
            left = np.repeat(rows,lengths)
            keep = left != right
            # query radius is rounded outward; retain the same <= convention
            # as the random-position spherical neighbour query.
            keep &= np.linalg.norm(vectors[left]-vectors[right],axis=1) <= limit
            _union_edges(parent,left[keep],right[keep])
    roots = _roots(parent)
    sizes = np.bincount(roots,minlength=n)
    n_blend = sizes[roots].astype(np.int64)
    group_index = np.full(n,-1,dtype=np.int64)
    order = np.argsort(roots,kind='stable')
    breaks = np.flatnonzero(np.diff(roots[order]))+1
    members = tuple(rows for rows in np.split(order,breaks) if rows.size>1)
    for group,rows in enumerate(members):
        group_index[rows] = group
        rows.flags.writeable = False
    ids = ids.copy()
    for value in (ids,group_index,n_blend):
        value.flags.writeable = False
    return DetectedGroups(ids,group_index,members,n_blend)
