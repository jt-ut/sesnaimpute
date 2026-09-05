"""The single read path from any granule to a region's source order.

`region_slice` reads a region's own rows from the granule map once (cached
per process). `per_source` maps a product at any of the six granules
(IMPLEMENTATION.md section 1) to that region's catalogue-row order,
dispatching on the product's own `GRANULE` root attribute -- never a
registry.
"""

import os

import h5py
import numpy as np

from sesnaimpute.config import product_path

_REGION_SLICE_CACHE = {}
_PRIMARY_REGION_CACHE = {}


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def region_slice(config, region):
    """The region's own source rows -- catalogue row, HPX256/512 pixel,
    HPX256_ROW, SIGHTLINE_ID -- read once from the granule map and cached
    for the life of the process."""
    key = (id(config), region)
    cached = _REGION_SLICE_CACHE.get(key)
    if cached is not None:
        return cached
    path = product_path(config, "granules", "sesna", "granule-map", "source")
    with h5py.File(path, "r") as f:
        names = [_text(v) for v in f["region/REGION"][:]]
        if region not in names:
            raise ValueError("region_slice: region %r is absent from granule map %s" % (region, path))
        i = names.index(region)
        offset = int(f["region/SOURCE_ROW_OFFSET"][i])
        n = int(f["region/N_SOURCE_ROWS"][i])
        sl = slice(offset, offset + n)
        out = {
            "n_sources": n,
            "catalog_row": np.asarray(f["source/CATALOG_ROW"][sl]),
            "hpx_pix_256": np.asarray(f["source/HPX_PIX_256"][sl]),
            "hpx_pix_512": np.asarray(f["source/HPX_PIX_512"][sl]),
            "hpx256_row": np.asarray(f["source/HPX256_ROW"][sl]),
            "sightline_id": np.asarray(f["source/SIGHTLINE_ID"][sl]),
        }
    _REGION_SLICE_CACHE[key] = out
    return out


def _source_row_key(handle, row_slice, n_sources, path):
    """The file's own row-identity column, if it carries one, verified to
    be exactly a permutation of range(n_sources); `None` when absent, in
    which case the file is trusted to already be in catalogue-row order."""
    for name in ("CATALOG_ROW", "ROWINDEX", "ROW"):
        if name not in handle:
            continue
        key = np.asarray(handle[name][row_slice], dtype=np.int64)
        if key.shape[0] != n_sources:
            raise ValueError("per_source: %s's %r has %d rows, region has %d catalogued sources"
                             % (path, name, key.shape[0], n_sources))
        if not np.array_equal(np.sort(key), np.arange(n_sources)):
            raise ValueError("per_source: %s's %r is not a permutation of range(%d) -- "
                             "row alignment cannot be assumed" % (path, name, n_sources))
        return np.argsort(key)
    return None


def _read_source_granule(path, region, columns, n_sources):
    with h5py.File(path, "r") as f:
        # a source-granule family's columns are flat top-level datasets;
        # the granule map itself is the one exception, keeping its per-
        # source columns in a "source" group and, since it is survey-wide
        # rather than per-region, its own region row-range in "region"
        if "source" in f and all(c not in f for c in columns):
            grp = f["source"]
            names = [_text(v) for v in f["region/REGION"][:]]
            i = names.index(region)
            row_slice = slice(int(f["region/SOURCE_ROW_OFFSET"][i]),
                              int(f["region/SOURCE_ROW_OFFSET"][i]) + int(f["region/N_SOURCE_ROWS"][i]))
        else:
            grp, row_slice = f, slice(None)
        order = _source_row_key(grp, row_slice, n_sources, path)
        missing = [c for c in columns if c not in grp]
        if missing:
            raise ValueError("per_source: %s has no column(s) %s" % (path, missing))
        out = {}
        for col in columns:
            arr = np.asarray(grp[col][row_slice])
            if arr.shape[0] != n_sources:
                raise ValueError("per_source: %s's %r has %d rows, region has %d catalogued sources"
                                 % (path, col, arr.shape[0], n_sources))
            out[col] = arr[order] if order is not None else arr
    return out


def _read_keyed_granule(path, key_name, columns, source_grain, missing_value):
    with h5py.File(path, "r") as f:
        if key_name not in f:
            raise ValueError("per_source: %s has no grain key %r" % (path, key_name))
        keys = np.asarray(f[key_name][:], dtype=np.int64)
        order = np.argsort(keys)
        sorted_keys = keys[order]
        if sorted_keys.size and np.any(sorted_keys[1:] == sorted_keys[:-1]):
            raise ValueError("per_source: %s's grain key %r is not unique -- "
                             "not one row per occupied grain" % (path, key_name))
        src = np.asarray(source_grain, dtype=np.int64)
        loc = np.searchsorted(sorted_keys, src)
        if sorted_keys.size:
            capped = np.minimum(loc, sorted_keys.size - 1)
            valid = (loc < sorted_keys.size) & (sorted_keys[capped] == src)
        else:
            capped = np.zeros(src.shape, dtype=np.intp)
            valid = np.zeros(src.shape, dtype=bool)
        if not np.all(valid):
            absent = np.unique(src[~valid])
            if missing_value is None:
                raise ValueError(
                    "per_source: %d of %d source(s) have a grain %r-value absent from %s "
                    "-- e.g. %s; pass missing= to fill instead of refusing"
                    % (int(np.count_nonzero(~valid)), src.size, key_name, path, absent[:5].tolist()))
        file_rows = order[capped] if sorted_keys.size else np.zeros(src.shape, dtype=np.intp)
        missing = [c for c in columns if c not in f]
        if missing:
            raise ValueError("per_source: %s has no column(s) %s" % (path, missing))
        out = {}
        for col in columns:
            arr = np.asarray(f[col][:])
            vals = arr[file_rows] if arr.size else np.empty(src.shape, dtype=arr.dtype)
            if not np.all(valid):
                vals = vals.astype(vals.dtype if vals.dtype.kind in "fc" else object, copy=True)
                vals[~valid] = missing_value
            out[col] = vals
    return out


def _read_region_granule(path, region, columns, n_sources):
    with h5py.File(path, "r") as f:
        if "REGION" not in f:
            raise ValueError("per_source: %s has no REGION column" % path)
        names = [_text(v) for v in f["REGION"][:]]
        if region not in names:
            raise ValueError("per_source: %s has no row for region %r" % (path, region))
        idx = names.index(region)
        missing = [c for c in columns if c not in f]
        if missing:
            raise ValueError("per_source: %s has no column(s) %s" % (path, missing))
        out = {}
        for col in columns:
            value = np.asarray(f[col][idx])
            out[col] = np.full(n_sources, value, dtype=value.dtype)
    return out


def _tile_membership(config):
    path = product_path(config, "bms", "anchors", "tile-membership", "hpx512")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "per_source: tile membership %s does not exist yet -- run the STAR anchor "
            "stage's RUNBOOK line first" % path)
    with h5py.File(path, "r") as f:
        pix = np.asarray(f["HPX_PIX_512"][:], dtype=np.int64)
        tile = np.asarray(f["TILE"][:])
    order = np.argsort(pix)
    return pix[order], tile[order]


def per_source(config, region, path, columns, missing=None, granule=None):
    """`{column: array}`, one row per catalogued source of `region`, in
    catalogue-row order, for `path` -- the join every consumer of a
    granule product performs. The product's granule is read from its own
    `GRANULE` root attribute; `granule=` is an override for reading an old
    product that carries no such attribute (documented at each call site
    that needs it), and is refused if it disagrees with an attribute the
    file does carry."""
    columns = list(columns)
    rs = region_slice(config, region)
    n_sources = rs["n_sources"]
    with h5py.File(path, "r") as f:
        file_granule = f.attrs.get("GRANULE")
    if isinstance(file_granule, bytes):
        file_granule = file_granule.decode("utf-8")
    if file_granule is not None and granule is not None and file_granule != granule:
        raise ValueError("per_source: %s declares GRANULE %r, caller asserted %r"
                         % (path, file_granule, granule))
    resolved = file_granule or granule
    if resolved is None:
        raise ValueError("per_source: %s carries no GRANULE root attribute; pass granule= "
                         "to read an old product that predates it" % path)

    if resolved == "source":
        return _read_source_granule(path, region, columns, n_sources)
    if resolved == "hpx512":
        return _read_keyed_granule(path, "HPX_PIX_512", columns, rs["hpx_pix_512"], missing)
    if resolved == "sightline":
        return _read_keyed_granule(path, "HPX_PIX_256", columns, rs["hpx_pix_256"], missing)
    if resolved == "tile":
        tile_pix, tile_of_pix = _tile_membership(config)
        loc = np.searchsorted(tile_pix, rs["hpx_pix_512"])
        capped = np.minimum(loc, tile_pix.size - 1) if tile_pix.size else loc
        valid = tile_pix.size and (tile_pix[capped] == rs["hpx_pix_512"])
        if not np.all(valid):
            raise ValueError("per_source: %d source(s) have an hpx512 pixel with no tile "
                             "in the STAR anchor tile-membership table"
                             % int(np.count_nonzero(~np.asarray(valid))))
        source_tile = tile_of_pix[capped]
        return _read_keyed_granule(path, "TILE", columns, source_tile, missing)
    if resolved == "region":
        return _read_region_granule(path, region, columns, n_sources)
    if resolved == "survey":
        with h5py.File(path, "r") as f:
            missing_cols = [c for c in columns if c not in f]
            if missing_cols:
                raise ValueError("per_source: %s has no column(s) %s" % (path, missing_cols))
            out = {}
            for col in columns:
                value = np.asarray(f[col][()])
                out[col] = np.full(n_sources, value, dtype=value.dtype)
        return out
    raise ValueError("per_source: %s declares unsupported GRANULE %r" % (path, resolved))


def primary_region_for_pixel(config, pix256):
    """The lowest-`REGION_CODE` region name at each nside-256 grain in
    `pix256`, from the granule map's own association table; `""` for a
    grain the map does not associate with any region."""
    path = product_path(config, "granules", "sesna", "granule-map", "source")
    cached = _PRIMARY_REGION_CACHE.get(path)
    if cached is None:
        with h5py.File(path, "r") as f:
            assoc = f["association/region_healpix256"]
            assoc_pix = np.asarray(assoc["HPX_PIX_256"][:], dtype=np.int64)
            assoc_code = np.asarray(assoc["REGION_CODE"][:], dtype=np.int64)
            reg_codes = np.asarray(f["region/REGION_CODE"][:], dtype=np.int64)
            reg_names = [_text(v) for v in f["region/REGION"][:]]
        code_to_name = dict(zip(reg_codes.tolist(), reg_names))
        order = np.lexsort((assoc_code, assoc_pix))
        pix_sorted, code_sorted = assoc_pix[order], assoc_code[order]
        first = np.concatenate(([True], pix_sorted[1:] != pix_sorted[:-1]))
        uniq_pix, uniq_code = pix_sorted[first], code_sorted[first]
        cached = (uniq_pix, uniq_code, code_to_name)
        _PRIMARY_REGION_CACHE[path] = cached
    uniq_pix, uniq_code, code_to_name = cached

    pix = np.asarray(pix256, dtype=np.int64)
    if uniq_pix.size == 0:
        matched = np.zeros(pix.shape, dtype=bool)
        loc = np.zeros(pix.shape, dtype=np.int64)
    else:
        loc = np.clip(np.searchsorted(uniq_pix, pix), 0, uniq_pix.size - 1)
        matched = uniq_pix[loc] == pix
    codes = np.where(matched, uniq_code[loc], -1)
    return np.array([code_to_name.get(int(c), "") for c in codes], dtype=object)
