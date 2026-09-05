"""2MASS PSC anchor source rows per region (SPEC_PRIORS.md section 2.1,
the STAR anchor: "2MASS PSC at Ks < 14.3, clean photometry flag ...
Half-magnitude histograms per nside-512 pixel", read against
`04_star_family.md` section B's `read_anchor_counts` row and the old
`fetch_external.twomass_source_counts.build.healpix512` module's own
Ks < 14.3 exact cut / Ks < 15.5, half-mag histogram design).

Download never computes -- but unlike the Gaia module beside it, this one
cannot produce a per-pixel, per-bin histogram inside the TAP query
itself: IRSA's `fp_psc` TAP service is Oracle-backed and has no HEALPix
function (`ivo_healpix_index` was tried against it directly and IRSA
returned "ORA-00904: IVO_HEALPIX_INDEX: invalid identifier" -- checked
before this module was written; the table also carries no precomputed
HEALPix column, only an HTM-20 index, `htm20`, on an unrelated
tessellation). So this module fetches verbatim per-source rows -- clean
photometry only, `Ks` and Galactic position -- and the nside-512 binning
that pairs it with the Gaia module's product happens in
`sesnaimpute.sky.derived.twomass_counts`, the one `derive` step in this
pair that legitimately computes.

Query, per region (the exact text sent, `{l0}`/`{l1}`/`{b0}`/`{b1}` a
padded Galactic bounding box around the region's occupied nside-512
pixels, from the granule map)::

    SELECT glon, glat, k_m FROM fp_psc
    WHERE k_m IS NOT NULL AND k_m < 15.5 AND cc_flg = '000'
      AND glon BETWEEN {l0} AND {l1} AND glat BETWEEN {b0} AND {b1}

`k_m < 15.5` matches the old module's `KS_HIST_LIMIT`, the top of the
half-magnitude histogram (9.0-15.5); `cc_flg = '000'` is 2MASS's own
clean-photometry flag, the spec's "clean photometry flag" clause, applied
here rather than downstream since it is a selection on 2MASS's own
external population, not a SESNA measurement. `glon`/`glat` are fp_psc's
native Galactic coordinates -- no frame conversion, so the derive step's
`healpy.ang2pix(512, glon, glat, nest=True, lonlat=True)` lands on this
project's own Galactic nside-512 grid directly.
"""

import os
import urllib.parse
import urllib.request

import healpy as hp
import numpy as np

from sesnaimpute import build as build_module
from sesnaimpute import regions as regions_module
from sesnaimpute.granules import access

IRSA_TAP_SYNC_URL = "https://irsa.ipac.caltech.edu/TAP/sync"
TABLE = "fp_psc"
NSIDE = 512
KS_HIST_LIMIT = 15.5  # old module's KS_HIST_LIMIT / SPEC_PRIORS.md 2.1 histogram top
CLEAN_CC_FLAG = "000"
BOX_PAD_DEG = 0.1  # more than half an hpx512 pixel's ~6.9' diagonal


def _lb_bounding_box(config, region, pad_deg=BOX_PAD_DEG):
    """A padded Galactic (l, b) box enclosing every nside-512 pixel the
    region's own catalogued sources occupy, per the granule map."""
    pix = np.unique(access.region_slice(config, region)["hpx_pix_512"])
    l_deg, b_deg = hp.pix2ang(NSIDE, pix, nest=True, lonlat=True)
    return (float(l_deg.min()) - pad_deg, float(l_deg.max()) + pad_deg,
            float(b_deg.min()) - pad_deg, float(b_deg.max()) + pad_deg)


def region_query(config, region):
    """The exact ADQL text this module sends for `region`."""
    l0, l1, b0, b1 = _lb_bounding_box(config, region)
    return (
        "SELECT glon, glat, k_m FROM "
        f"{TABLE} WHERE k_m IS NOT NULL AND k_m < {KS_HIST_LIMIT:.1f} "
        f"AND cc_flg = '{CLEAN_CC_FLAG}' "
        f"AND glon BETWEEN {l0:.6f} AND {l1:.6f} AND glat BETWEEN {b0:.6f} AND {b1:.6f}"
    )


def _tap_sync_csv(query, sync_url=IRSA_TAP_SYNC_URL, timeout=600):
    data = urllib.parse.urlencode({"QUERY": query, "FORMAT": "csv", "LANG": "ADQL"}).encode("utf-8")
    with urllib.request.urlopen(sync_url, data=data, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def build(config, regions=None):
    """Writes `sky/download/twomass_counts/counts_twomass_hpx512__<Region>.csv`
    for each requested region (default: all thirty), each the verbatim
    CSV response of one clean-photometry row query (see module docstring
    for why this one is per-source rows, not per-pixel counts)."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    dest_dir = f"{config.data_root}/sky/download/twomass_counts"
    os.makedirs(dest_dir, exist_ok=True)
    for region in regions:
        query = region_query(config, region)
        text = _tap_sync_csv(query)
        dest_path = f"{dest_dir}/counts_twomass_hpx512__{region}.csv"
        with open(dest_path, "w") as f:
            f.write(text)
        n_rows = max(0, text.count("\n") - 1)
        print(f"twomass_counts build: {region} -> {dest_path}: {n_rows} rows")


if __name__ == "__main__":
    build_module.run(build)
