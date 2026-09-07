"""Builds the survey-wide granule map (IMPLEMENTATION.md section 1).

Stores, once per catalogued source, its nside-256 and nside-512 galactic
NESTED HEALPix pixel, its sightline row (the row into its region's own
sorted source-bearing nside-256 pixels), its survey-wide sightline id, and
its region. A nside-256 pixel is admitted if it carries at least one
SESNA source (SPEC_PRIORS.md section 0.2's admission definition); every
nside-512 child of an admitted nside-256 pixel is admitted too. A pixel
with positive Spitzer mosaic support but no catalogued source is no
longer admitted (owner, 2026-09-06) -- mosaic-support-only pixels fed no
source-keyed product, so admitting them only inflated the granule map's
own footprint; each pixel's own mosaic-support flag is kept regardless
(`SESNA_MOSAIC_SUPPORTED`), for a reader that still wants it.
"""

import os

import h5py
import healpy as hp
import numpy as np

from sesnaimpute import build as build_module
from sesnaimpute import progress as progress_module
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


def _read_existing_regions(path):
    """Every region's own per-source ra/dec/l/b, name and pixel columns,
    sliced back out of a previously-written granule map by that region's
    own `SOURCE_ROW_OFFSET`/`N_SOURCE_ROWS`. Returns `None` if the file is
    missing or unreadable, so the caller falls back to a fresh build."""
    if not os.path.exists(path):
        return None
    try:
        with h5py.File(path, "r") as f:
            names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in f["region/REGION"][:]]
            offsets = f["region/SOURCE_ROW_OFFSET"][:]
            counts = f["region/N_SOURCE_ROWS"][:]
            src = f["source"]
            cols = {c: src[c][:] for c in
                    ("RA_DEG", "DEC_DEG", "GAL_L_DEG", "GAL_B_DEG", "NAME", "HPX_PIX_256", "HPX_PIX_512")}
        out = {}
        for name, off, n in zip(names, offsets, counts):
            sl = slice(int(off), int(off) + int(n))
            out[name] = dict(ra=cols["RA_DEG"][sl], dec=cols["DEC_DEG"][sl],
                              gl=cols["GAL_L_DEG"][sl], gb=cols["GAL_B_DEG"][sl],
                              name=cols["NAME"][sl], pix256=cols["HPX_PIX_256"][sl],
                              pix512=cols["HPX_PIX_512"][sl])
        return out
    except Exception:
        return None


def build(config, regions=None):
    """Writes the one survey-wide granule map. Every region is written
    every call -- the healpix-admission and association tables are joint
    across the whole survey, so they cannot be built from a subset of
    regions alone. `--regions` instead picks which regions' source rows
    are re-derived from the curated catalogue this call: the rest are
    carried over unchanged from the granule map already on disk (the same
    in-place convention the region-axis tables use), so a subset run
    still does no catalogue work for the regions it leaves out."""
    ordered_names = [r.name for r in regions_module.REGIONS]
    out_path = product_path(config, "granules", "sesna", "granule-map", "source")

    if regions is not None:
        wanted = set(regions)
        carried = _read_existing_regions(out_path)
        if carried is None:
            print("granules.build: --regions given but no existing granule map at "
                  "%s to carry the rest over from -- building every region fresh" % out_path,
                  flush=True)
    else:
        wanted = set(ordered_names)
        carried = None

    with progress_module.Stage("granules.build") as st:
        per_region = []
        sightline_offset = 0
        source_offset = 0
        n_regions = len(ordered_names)
        for code, region in enumerate(ordered_names):
            if region in wanted or carried is None or region not in carried:
                arrays = _region_arrays(config, region)
            else:
                arrays = carried[region]
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
            st.tick(code + 1, n_regions, "regions")

        n_sources = source_offset
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

            # admission (owner, 2026-09-06): a pixel carries at least one
            # SESNA source, full stop; every nside-512 child of an admitted
            # nside-256 pixel is admitted too. `direct256`/`direct512` (the
            # mosaic-support-only pixels) no longer widen admission -- they
            # are kept only to flag `SESNA_MOSAIC_SUPPORTED` below and to
            # report the before/after admission counts.
            direct256 = np.unique(np.concatenate([r["supported256"] for r in per_region])) \
                if per_region else np.empty(0, dtype=np.int64)
            all_source_pix256 = np.concatenate([r["pix256"] for r in per_region]) if n_sources else np.empty(0, "i8")
            source256, counts256 = np.unique(all_source_pix256, return_counts=True)
            healpix256_before = np.union1d(source256, direct256)
            healpix256 = source256
            healpix512 = (healpix256[:, None] * 4 + np.arange(4, dtype=np.int64)[None, :]).reshape(-1) \
                if healpix256.size else np.empty(0, dtype=np.int64)
            healpix512_before = (healpix256_before[:, None] * 4 + np.arange(4, dtype=np.int64)[None, :]).reshape(-1) \
                if healpix256_before.size else np.empty(0, dtype=np.int64)
            direct512 = np.unique(np.concatenate([r["supported512"] for r in per_region])) \
                if per_region else np.empty(0, dtype=np.int64)
            all_source_pix512 = np.concatenate([r["pix512"] for r in per_region]) if n_sources else np.empty(0, "i8")
            source512, counts512 = np.unique(all_source_pix512, return_counts=True)

            print("granules.build: admission (SESNA-source pixels only): "
                  "nside-256 %d -> %d (dropped %d mosaic-support-only), "
                  "nside-512 %d -> %d (dropped %d)"
                  % (healpix256_before.size, healpix256.size, healpix256_before.size - healpix256.size,
                     healpix512_before.size, healpix512.size, healpix512_before.size - healpix512.size))
            for r in per_region:
                if r["region"] != "NGC 7129":
                    continue
                before256 = np.union1d(r["source_pix256"], r["supported256"])
                before512 = (before256[:, None] * 4 + np.arange(4, dtype=np.int64)[None, :]).reshape(-1) \
                    if before256.size else np.empty(0, dtype=np.int64)
                after512 = (r["source_pix256"][:, None] * 4 + np.arange(4, dtype=np.int64)[None, :]).reshape(-1) \
                    if r["source_pix256"].size else np.empty(0, dtype=np.int64)
                print("granules.build: NGC 7129 admission: nside-256 %d -> %d, "
                      "nside-512 %d -> %d"
                      % (before256.size, r["source_pix256"].size, before512.size, after512.size))

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

        st.done(out_path, regions=n_regions, sources=n_sources,
                healpix256=int(healpix256.size), healpix512=int(healpix512.size))


if __name__ == "__main__":
    build_module.run(build)
