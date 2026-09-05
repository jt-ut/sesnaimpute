"""Gaia DR3 anchor counts per nside-512 HEALPix pixel (SPEC_PRIORS.md
section 2.1, the STAR anchor: "Gaia DR3 at G < 19 ... half-magnitude
histograms per nside-512 pixel" -- read exactly as G < 19 with the wider
G < 21 histogram of `04_star_family.md` section B's `read_anchor_counts`
row and the old `fetch_external.gaia_source_counts.build.healpix512`
module's own bins).

Download never computes: the per-pixel, per-magnitude-bin counts are
produced by the TAP service's own GROUP BY, and this module writes the
returned table verbatim, one CSV per region. The pixel scheme is the
project's own nside-512 galactic NESTED grid ("hpx512",
IMPLEMENTATION.md section 1) -- level 9 of the same hierarchy whose
level 12 is embedded in a Gaia `source_id` (bits 36-63; not used here,
since that encoding is ICRS-frame and would not match this pixel grid,
which is Galactic). Instead each row is binned at query time by the
IVOA ADQL geometry function `ivo_healpix_index(order, lon, lat)`, order
9 giving nside 512 NESTED, applied to Gaia's own `l`/`b` columns so the
index lands on this project's Galactic grid directly -- matches
`healpy.ang2pix(512, l, b, nest=True, lonlat=True)` exactly.

Query, per region (the exact text sent, `{l0}`/`{l1}`/`{b0}`/`{b1}` a
padded Galactic bounding box around the region's occupied nside-512
pixels, from the granule map). Aliases are lower-case because ESA's TAP
layer compiles to Postgres SQL that quotes a mixed-case `SELECT` alias
but not the matching `GROUP BY` reference, so the two fail to resolve
against each other unless both are already lower-case::

    SELECT ivo_healpix_index(9, l, b) AS hpx_pix_512,
           FLOOR(phot_g_mean_mag) AS g_bin_lo,
           COUNT(*) AS n
    FROM gaiadr3.gaia_source
    WHERE phot_g_mean_mag IS NOT NULL AND phot_g_mean_mag < 21
      AND l BETWEEN {l0} AND {l1} AND b BETWEEN {b0} AND {b1}
    GROUP BY hpx_pix_512, g_bin_lo

sent to ESA's own Gaia archive TAP sync endpoint (not the old module's
NOIRLAB Data Lab mirror: ESA's TAP layer is the one carrying
`ivo_healpix_index`, letting the histogram be built in one query instead
of the old module's per-HPX64-tile row fetch and client-side binning).
`g_bin_lo` runs 10..20 (`phot_g_mean_mag < 21` floors to at most 20),
the same eleven 1-mag bins as the old module's `G_MAG_EDGES =
arange(10, 22, 1)`; a row's bin index is `g_bin_lo - 10`. The returned
pixel set is a superset of the region's occupied pixels (a bounding
box, not an exact polygon); `sesnaimpute.sky.derived.gaia_counts`
restricts it to the granule map's exact set.
"""

import os
import urllib.parse
import urllib.request

import healpy as hp
import numpy as np

from sesnaimpute import build as build_module
from sesnaimpute import regions as regions_module
from sesnaimpute.granules import access

GAIA_TAP_SYNC_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
GAIA_TABLE = "gaiadr3.gaia_source"
NSIDE = 512
G_LIMIT = 21.0  # SPEC_PRIORS.md section 2.1 / old module's GAIA_G_LIMIT
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
        "SELECT ivo_healpix_index(9, l, b) AS hpx_pix_512, "
        "FLOOR(phot_g_mean_mag) AS g_bin_lo, COUNT(*) AS n "
        f"FROM {GAIA_TABLE} "
        f"WHERE phot_g_mean_mag IS NOT NULL AND phot_g_mean_mag < {G_LIMIT:.1f} "
        f"AND l BETWEEN {l0:.6f} AND {l1:.6f} AND b BETWEEN {b0:.6f} AND {b1:.6f} "
        "GROUP BY hpx_pix_512, g_bin_lo"
    )


def _tap_sync_csv(query, sync_url=GAIA_TAP_SYNC_URL, timeout=600):
    data = urllib.parse.urlencode({"QUERY": query, "FORMAT": "csv", "LANG": "ADQL-2.0"}).encode("utf-8")
    with urllib.request.urlopen(sync_url, data=data, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def build(config, regions=None):
    """Writes `sky/download/gaia_counts/counts_gaia_hpx512__<Region>.csv`
    for each requested region (default: all thirty), each the verbatim
    CSV response of one grouped TAP query."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    dest_dir = f"{config.data_root}/sky/download/gaia_counts"
    os.makedirs(dest_dir, exist_ok=True)
    for region in regions:
        dest_path = f"{dest_dir}/counts_gaia_hpx512__{region}.csv"
        if os.path.exists(dest_path):
            print(f"gaia_counts build: {dest_path} present, skipped")
            continue
        query = region_query(config, region)
        text = _tap_sync_csv(query)
        dest_path = f"{dest_dir}/counts_gaia_hpx512__{region}.csv"
        with open(dest_path, "w") as f:
            f.write(text)
        n_rows = max(0, text.count("\n") - 1)
        print(f"gaia_counts build: {region} -> {dest_path}: {n_rows} rows")


if __name__ == "__main__":
    build_module.run(build)
