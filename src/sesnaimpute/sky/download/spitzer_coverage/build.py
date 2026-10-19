"""SESNA's own native Spitzer IRAC/MIPS mosaic coverage masks, per field,
read straight from the re-downloaded release (the retired package's earlier
copy of this file was silently truncated at 25 files per region and is
missing MIPS entirely, and every IRAC field, in six regions).

`build(config, regions=None, source=...)` inventories `source` -- a
directory holding either the release's 87 zips
(`<Region>-<timestamp>-1-NNN.zip`, members
`<Region>/<Region>_<IRAC|MIPS>_Field<n>_<i1|i2|i3|i4|m1>_<cov|flux>.fits`)
or the same `*_cov.fits` files loose, a trailing `-NNN` duplicate-download
suffix stripped -- and, for each requested region, reads every field's
primary HDU (only `*_cov.fits`; `*_flux.fits` is never opened here), forms
the mask `sky.derived.coverage._field_masks` reads
(`np.isfinite(data) & (data > 0)`), and writes it with the WCS cards that
reader needs plus the field's own covered area and source filename. The
87 zips total 69.5 GB uncompressed; none of those bytes are kept -- the
written file is the product, at `sky/download/spitzer_coverage/
spitzer_coverage_masks.hdf5` (the path `sky.derived.coverage._masks_path`
names).

Before writing a region, its own (field, band) inventory is checked
against the curated catalogue's own `ORIGIN_FNU` (a measured detection in
a band this release carries no mask for fails the build) and against
itself (every IRAC field must carry the same four bands as every other
IRAC field; MIPS fields are their own numbering, unconstrained).
"""

import io
import os
import re
import zipfile

import h5py
import numpy as np
from astropy.io import fits

from sesnaimpute import build as build_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run

#: the five Spitzer bands these masks carry (the three 2MASS bands are
#: all-sky and are never part of this product).
BANDS = ("I1", "I2", "I3", "I4", "M1")

#: this product's own registry stem (attrs_registry.py): the file's own
#: basename with the extension removed. Its field-mask datasets are named
#: `field_Field<n>`, one varying name per field rather than the fixed name
#: set the registry convention assumes, so every one of them shares the
#: single "FIELD_MASK" reading below instead of being keyed by its own name.
_STEM = "spitzer_coverage_masks"

_BAND_KEY = {"i1": "I1", "i2": "I2", "i3": "I3", "i4": "I4", "m1": "M1"}

#: `<Region>_<IRAC|MIPS>_Field<n>_<i1|i2|i3|i4|m1>_cov[-NNN].fits`: the
#: region name is written verbatim (it may itself hold a space or a comma,
#: e.g. "Cygnus X", "GGD4, CB34") immediately followed by the instrument
#: token, which the region names never themselves contain.
_COV_RE = re.compile(
    r"^(?P<region>.+)_(?:IRAC|MIPS)_Field(?P<field>\d+)_(?P<band>i1|i2|i3|i4|m1)_cov(?:-\d+)?\.fits$")


def _is_cov_name(basename):
    return basename.endswith("_cov.fits") or re.search(r"_cov-\d+\.fits$", basename) is not None


def _parse_cov_name(basename):
    m = _COV_RE.match(basename)
    if not m:
        raise ValueError(
            f"sky.download.spitzer_coverage: filename does not parse as a region/field/band "
            f"coverage FITS: {basename!r}")
    return m.group("region"), _BAND_KEY[m.group("band")], int(m.group("field"))


def _inventory(source):
    """`{region: [record, ...]}` for every `*_cov.fits` found loose under
    `source` or as a member of any `*.zip` under it. Each record is
    `dict(band=, field=, filename=, bytes=, open=<callable taking no
    arguments and returning a path or a seekable in-memory stream
    astropy.io.fits.open can read>)`."""
    by_region = {}
    for dirpath, _dirnames, filenames in os.walk(source):
        for fn in filenames:
            if fn.endswith(".zip"):
                zpath = os.path.join(dirpath, fn)
                with zipfile.ZipFile(zpath) as zf:
                    for info in zf.infolist():
                        base = os.path.basename(info.filename)
                        if not _is_cov_name(base):
                            continue
                        region, band, field = _parse_cov_name(base)
                        member = info.filename

                        def _opener(zpath=zpath, member=member):
                            with zipfile.ZipFile(zpath) as zf2:
                                return io.BytesIO(zf2.read(member))

                        by_region.setdefault(region, []).append(
                            dict(band=band, field=field, filename=base,
                                 bytes=info.file_size, open=_opener))
            elif _is_cov_name(fn):
                region, band, field = _parse_cov_name(fn)
                fpath = os.path.join(dirpath, fn)
                by_region.setdefault(region, []).append(
                    dict(band=band, field=field, filename=fn,
                         bytes=os.path.getsize(fpath), open=lambda fpath=fpath: fpath))
    return by_region


def _read_mask(record):
    """Reads one field's `*_cov.fits` primary HDU (one field in memory at a
    time; the handle `record["open"]()` returns is released at the end of
    this call) and returns `(mask, wcs_attrs, area_deg2)`: the mask
    `sky.derived.coverage._field_masks` reads, the WCS header cards that
    reader needs, and the field's own covered solid angle (covered pixel
    count times the CD-matrix pixel area)."""
    handle = record["open"]()
    with fits.open(handle, memmap=False) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        mask = np.isfinite(data) & (data > 0)
        pixel_area_deg2 = abs(header["CD1_1"] * header["CD2_2"] - header["CD1_2"] * header["CD2_1"])
        area_deg2 = float(mask.sum()) * pixel_area_deg2
        wcs_attrs = dict(
            CRPIX1=header["CRPIX1"], CRPIX2=header["CRPIX2"],
            CRVAL1=header["CRVAL1"], CRVAL2=header["CRVAL2"],
            CTYPE1=str(header["CTYPE1"]), CTYPE2=str(header["CTYPE2"]),
            CD1_1=header["CD1_1"], CD1_2=header["CD1_2"],
            CD2_1=header["CD2_1"], CD2_2=header["CD2_2"],
            # RADESYS is the modern WCS keyword; these mosaics' own FITS
            # headers carry only its deprecated alias RADECSYS.
            RADESYS=header.get("RADESYS", header.get("RADECSYS", "")),
            EQUINOX=header.get("EQUINOX", 0.0))
    return mask, wcs_attrs, area_deg2


def _check_rectangular_and_catalog(config, region, records):
    """Raises before anything is written: the IRAC (field, band) set must
    be rectangular (every IRAC field carries the same bands as every
    other), and the curated catalogue's `ORIGIN_FNU` (sources_sesna_source,
    band order J H Ks I1 I2 I3 I4 M1) must hold no measured detection
    (code 1) in a band this release carries no mask for. MIPS fields are
    their own numbering and are not checked for rectangularity. No FITS
    file is opened here."""
    irac_bands = sorted({r["band"] for r in records if r["band"] != "M1"})
    irac_fields = sorted({r["field"] for r in records if r["band"] != "M1"})
    for field in irac_fields:
        bands_here = sorted({r["band"] for r in records if r["band"] != "M1" and r["field"] == field})
        if bands_here != irac_bands:
            raise ValueError(
                f"sky.download.spitzer_coverage: {region} IRAC field {field} carries bands "
                f"{bands_here}, not every IRAC field's {irac_bands} -- the IRAC (field, band) "
                f"set must be rectangular")

    catalog_path = f"{config.data_root}/catalog/sesna/sources_sesna_source__{region}.hdf5"
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(
            f"sky.download.spitzer_coverage: no curated catalogue at {catalog_path} -- run "
            "sesnaimpute.catalog.curated first")
    with h5py.File(catalog_path, "r") as cf:
        cat_bands = [b.decode() if isinstance(b, bytes) else b for b in cf.attrs["BANDS"]]
        origin_fnu = cf["ORIGIN_FNU"][:]
    present_bands = {r["band"] for r in records}
    for band in BANDS:
        if band not in cat_bands:
            continue
        if np.any(origin_fnu[:, cat_bands.index(band)] == 1) and band not in present_bands:
            raise ValueError(
                f"sky.download.spitzer_coverage: {region}'s curated catalogue has a measured "
                f"{band} detection (ORIGIN_FNU == 1) but the release carries no {band} mask "
                f"for this region")


def _write_region(masks_group, fields_group, region, records):
    """Writes every field in `records` under `MASKS/<region>/<band>/
    field_Field<n>` and the region's own `FIELDS` inventory, one field in
    memory at a time, and prints the one completeness-check summary line
    (CODING_RULES_BMSTP.md rule 17): n IRAC fields, n MIPS fields, covered
    area per band. A region already present (a rerun on a region subset)
    is replaced whole, never appended to."""
    if region in masks_group:
        del masks_group[region]
    if region in fields_group:
        del fields_group[region]
    region_group = masks_group.create_group(region)
    names_out, bytes_out, field_out, band_out = [], [], [], []
    band_area = {b: 0.0 for b in BANDS}
    for rec in records:
        mask, wcs_attrs, area_deg2 = _read_mask(rec)
        band_group = region_group.require_group(rec["band"])
        dataset = build_module.write_dataset(
            band_group, f"field_Field{rec['field']}", mask, *REGISTRY[(_STEM, "FIELD_MASK")],
            compression="gzip", compression_opts=4, chunks=True)
        for key, value in wcs_attrs.items():
            dataset.attrs[key] = value
        dataset.attrs["area_deg2"] = area_deg2
        dataset.attrs["source_filename"] = rec["filename"]
        dataset.attrs["source_bytes"] = rec["bytes"]
        band_area[rec["band"]] += area_deg2
        names_out.append(rec["filename"])
        bytes_out.append(rec["bytes"])
        field_out.append(rec["field"])
        band_out.append(rec["band"])
        del mask

    region_fields_group = fields_group.create_group(region)
    build_module.write_dataset(region_fields_group, "NAME", np.array(names_out, dtype="S128"),
                                *REGISTRY[(_STEM, "NAME")])
    build_module.write_dataset(region_fields_group, "BYTES", np.array(bytes_out, dtype=np.int64),
                                *REGISTRY[(_STEM, "BYTES")])
    build_module.write_dataset(region_fields_group, "FIELD", np.array(field_out, dtype=np.int32),
                                *REGISTRY[(_STEM, "FIELD")])
    build_module.write_dataset(region_fields_group, "BAND", np.array(band_out, dtype="S8"),
                                *REGISTRY[(_STEM, "BAND")])

    n_irac = len({r["field"] for r in records if r["band"] != "M1"})
    n_mips = len({r["field"] for r in records if r["band"] == "M1"})
    print(f"sky.download.spitzer_coverage: {region}: {n_irac} IRAC fields, {n_mips} MIPS fields, "
          + ", ".join(f"{b} area={band_area[b]:.4g} deg2" for b in BANDS), flush=True)


def build(config, regions=None, source=None):
    """Writes `sky/download/spitzer_coverage/spitzer_coverage_masks.hdf5`
    from the native coverage FITS (loose or in Drive zips) at `source`.
    `regions` defaults to every region named in `source`'s own inventory;
    a region named but absent from `source` fails."""
    if not source:
        raise ValueError(
            "sky.download.spitzer_coverage.build: source= (a directory holding the release's "
            "loose *_cov.fits files or its Drive zips) is required")
    names = regions or [r.name for r in regions_module.REGIONS]
    inventory = _inventory(source)
    out_path = f"{config.data_root}/sky/download/spitzer_coverage/spitzer_coverage_masks.hdf5"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with progress_module.Stage("sky.download.spitzer_coverage") as st:
        with h5py.File(out_path, "a") as f:
            f.attrs["SOURCE"] = source
            masks_group = f.require_group("MASKS")
            fields_group = f.require_group("FIELDS")
            for i, name in enumerate(names):
                records = inventory.get(name)
                if not records:
                    raise FileNotFoundError(
                        f"sky.download.spitzer_coverage: region {name!r} requested but not "
                        f"found under {source}")
                _check_rectangular_and_catalog(config, name, records)
                _write_region(masks_group, fields_group, name, records)
                st.tick(i + 1, len(names), "regions")
        st.done(out_path, regions=len(names))


if __name__ == "__main__":
    import sys

    _argv = sys.argv[1:]
    _source = None
    if "--source" in _argv:
        _i = _argv.index("--source")
        _source = _argv[_i + 1]
        del _argv[_i:_i + 2]
    if not _source:
        raise SystemExit(
            "sesnaimpute.sky.download.spitzer_coverage.build: --source <path> is required (a "
            "directory holding the release's loose coverage FITS or its Drive zips)")
    sys.argv = [sys.argv[0]] + _argv
    run(lambda config, regions=None: build(config, regions=regions, source=_source))
