"""Builds the survey-wide granule map (IMPLEMENTATION.md section 1).

Stores, once per catalogued source, its nside-256 and nside-512 galactic
NESTED HEALPix pixel, its sightline row (the row into its region's own
sorted source-bearing nside-256 pixels), its survey-wide sightline id, and
its region. A nside-256 pixel is admitted if it holds a source or has
positive any-band Spitzer mosaic support; every nside-512 child of an
admitted nside-256 pixel is admitted too.
"""

import os

import h5py
import healpy as hp
import numpy as np

from sesnaimpute import build as build_module
from sesnaimpute import regions as regions_module
from sesnaimpute.config import product_path

NSIDE_256 = 256
NSIDE_512 = 512
CHUNK_ROWS = 500_000
NAME_DTYPE = "S32"


def _catalog_path(config, region):
    return product_path(config, "catalog", "sesna", "sources", "source", region=region)


def _iter_catalog(config, region, chunk_rows=CHUNK_ROWS):
    """Chunks of (ra, dec, gl, gb, name) in the curated catalogue's own row
    order. Falls back to the old pandas-HDFStore SEDFIT_INPUT file when the
    new curated catalogue has not landed yet."""
    path = _catalog_path(config, region)
    old_path = f"{config.data_root}/catalog/curated/{region}_SEDFIT_INPUT.hdf5"
    if not os.path.exists(path) and os.path.exists(old_path):
        path = old_path
    with h5py.File(path, "r") as f:
        if "RA_DEG" in f:
            n = f["RA_DEG"].shape[0]
            for start in range(0, n, chunk_rows):
                stop = min(start + chunk_rows, n)
                yield (np.asarray(f["RA_DEG"][start:stop], dtype=np.float64),
                       np.asarray(f["DEC_DEG"][start:stop], dtype=np.float64),
                       np.asarray(f["GAL_L_DEG"][start:stop], dtype=np.float64),
                       np.asarray(f["GAL_B_DEG"][start:stop], dtype=np.float64),
                       np.asarray(f["NAME"][start:stop]).astype(NAME_DTYPE))
        else:  # remove when catalog/curated lands
            co = np.asarray(f["COORDS/table"][:])
            name = np.asarray(f["ID/table"][:])["SESNA_NAME"]
            yield (co["RA"].astype(np.float64), co["DEC"].astype(np.float64),
                   co["L"].astype(np.float64), co["B"].astype(np.float64),
                   np.asarray(name).astype(NAME_DTYPE))


_COVERAGE_GRANULE = {256: "sightline", 512: "hpx512"}


def _supported_pixels(config, region, nside):
    """Sorted nside-`nside` galactic NESTED pixels with positive coverage
    in any Spitzer band for `region` (IMPLEMENTATION.md section 1,
    SPEC_PRIORS.md section 2.1), read from that region's own
    `coverage_spitzer_{sightline,hpx512}` product."""
    granule = _COVERAGE_GRANULE[nside]
    path = product_path(config, "sky/derived", "spitzer", "coverage", granule, region=region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "granules.build: no coverage product at %s -- run "
            "sesnaimpute.sky.derived.coverage first" % path)
    with h5py.File(path, "r") as f:
        pix = np.asarray(f["HPX_PIX"][:], dtype=np.int64)
        frac = np.asarray(f["FRAC"][:], dtype=np.float64)
    positive = np.any(frac > 0.0, axis=1) if frac.size else np.zeros(0, dtype=bool)
    return np.sort(pix[positive])


def _region_arrays(config, region):
    """One region's per-source ra/dec/l/b, name, and nside-256/512 pixel."""
    ra_ch, dec_ch, gl_ch, gb_ch, name_ch, p256_ch, p512_ch = [], [], [], [], [], [], []
    for ra, dec, gl, gb, name in _iter_catalog(config, region):
        # the survey's nested galactic pixelisation at the two granule resolutions
        p256_ch.append(hp.ang2pix(NSIDE_256, gl, gb, nest=True, lonlat=True).astype(np.int64))
        p512_ch.append(hp.ang2pix(NSIDE_512, gl, gb, nest=True, lonlat=True).astype(np.int64))
        ra_ch.append(ra); dec_ch.append(dec); gl_ch.append(gl); gb_ch.append(gb); name_ch.append(name)
    cat = lambda chunks, dtype: (np.concatenate(chunks) if chunks
                                  else np.empty(0, dtype=dtype))
    return dict(
        ra=cat(ra_ch, np.float64), dec=cat(dec_ch, np.float64),
        gl=cat(gl_ch, np.float64), gb=cat(gb_ch, np.float64),
        name=cat(name_ch, NAME_DTYPE),
        pix256=cat(p256_ch, np.int64), pix512=cat(p512_ch, np.int64))


def _region_dataset(group, name, data):
    group.create_dataset(name, data=data, compression="gzip", compression_opts=4)


def build(config, regions=None):
    """Writes the one survey-wide granule map."""
    ordered_names = [r.name for r in regions_module.REGIONS]
    wanted = set(regions) if regions is not None else set(ordered_names)
    selected = [name for name in ordered_names if name in wanted]

    per_region = []
    sightline_offset = 0
    source_offset = 0
    for code, region in enumerate(selected):
        arrays = _region_arrays(config, region)
        n = arrays["pix256"].size
        # a source-bearing sightline is the region's own sorted set of
        # nside-256 pixels holding at least one catalogued source
        source_pix256 = np.unique(arrays["pix256"])
        hpx256_row = np.searchsorted(source_pix256, arrays["pix256"])
        sightline_id = sightline_offset + hpx256_row
        supported256 = _supported_pixels(config, region, NSIDE_256)
        supported512 = _supported_pixels(config, region, NSIDE_512)
        per_region.append(dict(
            region=region, code=code, source_offset=source_offset, n=n,
            source_pix256=source_pix256, supported256=supported256, supported512=supported512,
            hpx256_row=hpx256_row, sightline_id=sightline_id, **arrays))
        sightline_offset += source_pix256.size
        source_offset += n

    n_sources = source_offset
    out_path = product_path(config, "granules", "sesna", "granule-map", "source")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "source"

        src = f.create_group("source")
        _region_dataset(src, "REGION_CODE", np.concatenate(
            [np.full(r["n"], r["code"], dtype=np.int16) for r in per_region]) if n_sources else np.empty(0, "i2"))
        _region_dataset(src, "CATALOG_ROW", np.concatenate(
            [np.arange(r["n"], dtype=np.int64) for r in per_region]) if n_sources else np.empty(0, "i8"))
        _region_dataset(src, "NAME", np.concatenate([r["name"] for r in per_region]) if n_sources else np.empty(0, NAME_DTYPE))
        for col in ("ra", "dec", "gl", "gb", "pix256", "pix512", "hpx256_row", "sightline_id"):
            out_name = {"ra": "RA_DEG", "dec": "DEC_DEG", "gl": "GAL_L_DEG", "gb": "GAL_B_DEG",
                        "pix256": "HPX_PIX_256", "pix512": "HPX_PIX_512",
                        "hpx256_row": "HPX256_ROW", "sightline_id": "SIGHTLINE_ID"}[col]
            dtype = {"ra": "f8", "dec": "f8", "gl": "f8", "gb": "f8", "pix256": "i8", "pix512": "i8",
                    "hpx256_row": "i4", "sightline_id": "i4"}[col]
            _region_dataset(src, out_name, np.concatenate([r[col] for r in per_region]) if n_sources
                           else np.empty(0, dtype))

        reg = f.create_group("region")
        _region_dataset(reg, "REGION_CODE", np.array([r["code"] for r in per_region], dtype=np.int16))
        _region_dataset(reg, "REGION", np.array([r["region"] for r in per_region], dtype="S64"))
        _region_dataset(reg, "SOURCE_ROW_OFFSET", np.array([r["source_offset"] for r in per_region], dtype=np.int64))
        _region_dataset(reg, "N_SOURCE_ROWS", np.array([r["n"] for r in per_region], dtype=np.int64))

        # admission: a pixel holds a source or has positive mosaic support;
        # every nside-512 child of an admitted nside-256 pixel is admitted
        direct256 = np.unique(np.concatenate([r["supported256"] for r in per_region])) \
            if per_region else np.empty(0, dtype=np.int64)
        all_source_pix256 = np.concatenate([r["pix256"] for r in per_region]) if n_sources else np.empty(0, "i8")
        source256, counts256 = np.unique(all_source_pix256, return_counts=True)
        healpix256 = np.union1d(source256, direct256)
        healpix512 = (healpix256[:, None] * 4 + np.arange(4, dtype=np.int64)[None, :]).reshape(-1) \
            if healpix256.size else np.empty(0, dtype=np.int64)
        direct512 = np.unique(np.concatenate([r["supported512"] for r in per_region])) \
            if per_region else np.empty(0, dtype=np.int64)
        all_source_pix512 = np.concatenate([r["pix512"] for r in per_region]) if n_sources else np.empty(0, "i8")
        source512, counts512 = np.unique(all_source_pix512, return_counts=True)

        n_rows256 = np.zeros(healpix256.shape, dtype=np.int64)
        loc = np.searchsorted(healpix256, source256)
        n_rows256[loc] = counts256
        n_rows512 = np.zeros(healpix512.shape, dtype=np.int64)
        if healpix512.size:
            loc512 = np.searchsorted(healpix512, source512)
            n_rows512[loc512] = counts512

        hp256 = f.create_group("healpix256")
        _region_dataset(hp256, "HPX_PIX_256", healpix256)
        _region_dataset(hp256, "N_SOURCE_ROWS", n_rows256)
        _region_dataset(hp256, "SESNA_MOSAIC_SUPPORTED", np.isin(healpix256, direct256))

        hp512 = f.create_group("healpix512")
        _region_dataset(hp512, "HPX_PIX_512", healpix512)
        _region_dataset(hp512, "HPX_PIX_256", healpix512 // 4 if healpix512.size else healpix512)
        _region_dataset(hp512, "N_SOURCE_ROWS", n_rows512)
        _region_dataset(hp512, "SESNA_MOSAIC_SUPPORTED", np.isin(healpix512, direct512))

        assoc = f.create_group("association")
        for level, healpix, key in ((256, "pix256", "HPX_PIX_256"), (512, "pix512", "HPX_PIX_512")):
            rows = []
            for r in per_region:
                pix = r["pix256"] if level == 256 else r["pix512"]
                supported = r["supported256"] if level == 256 else r["supported512"]
                uniq, cnt = np.unique(pix, return_counts=True)
                count_of = dict(zip(uniq.tolist(), cnt.tolist()))
                members = sorted(set(count_of) | set(supported.tolist()))
                for p in members:
                    rows.append((p, r["code"], count_of.get(p, 0), p in set(supported.tolist())))
            rows.sort(key=lambda row: (row[0], row[1]))
            group = assoc.create_group(f"region_healpix{level}")
            _region_dataset(group, key, np.array([row[0] for row in rows], dtype=np.int64))
            _region_dataset(group, "REGION_CODE", np.array([row[1] for row in rows], dtype=np.int16))
            _region_dataset(group, "N_SOURCE_ROWS", np.array([row[2] for row in rows], dtype=np.int64))
            _region_dataset(group, "SESNA_MOSAIC_SUPPORTED", np.array([row[3] for row in rows], dtype=bool))


if __name__ == "__main__":
    build_module.run(build)
