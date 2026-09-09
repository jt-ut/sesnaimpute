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
photometry only, `J`, `H`, `Ks`, the per-band quality flag and Galactic
position -- and the nside-512 binning that pairs it with the Gaia
module's product happens in `sesnaimpute.sky.derived.twomass_counts`, the
one `derive` step in this pair that legitimately computes. `j_m`/`h_m`/
`ph_qual` feed a second, independent view,
`sesnaimpute.sky.derived.twomass_column_scale` (SPEC_BMSTP_DRAFT.md
3.1): the near-infrared colour of the same clean background stars, at
the map's own nside-512 pixels.

Query, per region (the exact text sent, `{l0}`/`{l1}`/`{b0}`/`{b1}` a
padded Galactic bounding box around the region's occupied nside-512
pixels, from the granule map)::

    SELECT glon, glat, j_m, h_m, k_m, ph_qual FROM fp_psc
    WHERE k_m IS NOT NULL AND k_m < 15.5 AND cc_flg = '000'
      AND glon BETWEEN {l0} AND {l1} AND glat BETWEEN {b0} AND {b1}

`k_m < 15.5` matches the old module's `KS_HIST_LIMIT`, the top of the
half-magnitude histogram (9.0-15.5); `cc_flg = '000'` is 2MASS's own
clean-photometry flag, the spec's "clean photometry flag" clause, applied
here rather than downstream since it is a selection on 2MASS's own
external population, not a SESNA measurement. `glon`/`glat` are fp_psc's
native Galactic coordinates -- no frame conversion, so a derive step's
`healpy.ang2pix(512, glon, glat, nest=True, lonlat=True)` lands on this
project's own Galactic nside-512 grid directly. `j_m`/`h_m`/`ph_qual` are
carried verbatim alongside `k_m`, unfiltered here -- the query's `WHERE`
is unchanged -- and selected on downstream.

A file already on disk whose header lacks `j_m` predates this widened
query and is refetched; a file whose header already carries it is
skipped, as before.
"""

import os
import re
import urllib.parse
import urllib.request

import healpy as hp
import numpy as np

from sesnaimpute import build as build_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.granules import access

IRSA_TAP_SYNC_URL = "https://irsa.ipac.caltech.edu/TAP/sync"
TABLE = "fp_psc"
NSIDE = 512
KS_HIST_LIMIT = 15.5  # old module's KS_HIST_LIMIT / SPEC_PRIORS.md 2.1 histogram top
CLEAN_CC_FLAG = "000"
BOX_PAD_DEG = 0.1  # more than half an hpx512 pixel's ~6.9' diagonal
#: Every query's Galactic-latitude span is cut to at most this many
#: degrees, so no single sync query -- even the densest region's, whose
#: full-box query timed out server-side -- is large enough for IRSA's
#: synchronous query limit to reject or truncate it.
BOX_STRIP_DEG = 1.0
CSV_HEADER = "glon,glat,j_m,h_m,k_m,ph_qual"


def _lb_bounding_box(config, region, pad_deg=BOX_PAD_DEG):
    """A padded Galactic (l, b) box enclosing every nside-512 pixel the
    region's own catalogued sources occupy, per the granule map.
    Wrap-safe about the pixel set's own median longitude, so a region
    straddling the l = 0/360 seam (e.g. Pipe) reads as its true few-
    degree span rather than nearly the whole sky in longitude."""
    pix = np.unique(access.region_slice(config, region)["hpx_pix_512"])
    l_deg, b_deg = hp.pix2ang(NSIDE, pix, nest=True, lonlat=True)
    c = float(np.median(l_deg))
    l_deg = c + ((l_deg - c + 180.0) % 360.0 - 180.0)
    return (float(l_deg.min()) - pad_deg, float(l_deg.max()) + pad_deg,
            float(b_deg.min()) - pad_deg, float(b_deg.max()) + pad_deg)


def region_strip_boxes(config, region, pad_deg=BOX_PAD_DEG, strip_deg=BOX_STRIP_DEG):
    """`region`'s padded (l, b) box cut into Galactic-latitude strips at
    most `strip_deg` wide: `[(l0, l1, b0_i, b1_i), ...]`, full longitude
    span, one query per strip."""
    l0, l1, b0, b1 = _lb_bounding_box(config, region, pad_deg)
    n_strips = max(1, int(np.ceil((b1 - b0) / strip_deg)))
    edges = np.linspace(b0, b1, n_strips + 1)
    return [(l0, l1, float(edges[i]), float(edges[i + 1])) for i in range(n_strips)]


def _glon_clause(l0, l1):
    """The `glon` WHERE clause for one (l0, l1) span: a plain `BETWEEN`
    when it sits inside one 0-360 turn, else the OR of the two pieces
    the l = 0/360 seam splits it into (`_lb_bounding_box` reports a
    wrapped span as `l0 < 0` or `l1 > 360`; `fp_psc`'s own `glon` is
    stored in [0, 360))."""
    if l0 >= 0.0 and l1 <= 360.0:
        return f"glon BETWEEN {l0:.6f} AND {l1:.6f}"
    lo, hi = l0 % 360.0, l1 % 360.0
    return f"(glon >= {lo:.6f} OR glon <= {hi:.6f})"


def strip_query(l0, l1, b0, b1):
    """The exact ADQL text for one (l0, l1, b0, b1) query box."""
    return (
        "SELECT glon, glat, j_m, h_m, k_m, ph_qual FROM "
        f"{TABLE} WHERE k_m IS NOT NULL AND k_m < {KS_HIST_LIMIT:.1f} "
        f"AND cc_flg = '{CLEAN_CC_FLAG}' "
        f"AND {_glon_clause(l0, l1)} AND glat BETWEEN {b0:.6f} AND {b1:.6f}"
    )


def _tap_sync_csv(query, sync_url=IRSA_TAP_SYNC_URL, timeout=600):
    data = urllib.parse.urlencode({"QUERY": query, "FORMAT": "csv", "LANG": "ADQL"}).encode("utf-8")
    with urllib.request.urlopen(sync_url, data=data, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _service_message(text):
    """A one-line description of what IRSA sent back instead of the
    `fp_psc` CSV: the VOTable `INFO` error text a server-side failure
    (e.g. a sync-query timeout) carries, else the response's own first
    line."""
    m = re.search(r'<INFO[^>]*name="QUERY_STATUS"[^>]*value="ERROR"[^>]*>(.*?)</INFO>',
                  text, re.S)
    if m:
        return m.group(1).strip()
    first = text.split("\n", 1)[0].strip()
    return first if first else "<empty response>"


def _strip_rows(config, region, l0, l1, b0, b1):
    """One strip's data rows (header stripped): raises if the response's
    first line is not the `fp_psc` CSV header, quoting IRSA's own
    message."""
    text = _tap_sync_csv(strip_query(l0, l1, b0, b1))
    lines = text.splitlines()
    if not lines or lines[0].strip() != CSV_HEADER:
        raise ValueError(
            "twomass_counts.build: region %r glat strip [%.3f, %.3f] -- IRSA "
            "did not return the fp_psc CSV header, it said: %s"
            % (region, b0, b1, _service_message(text)))
    return lines[1:]


def _header_line(path):
    with open(path, "r") as f:
        return f.readline().strip()


def _needs_refetch(dest_path):
    """A file predates the widened query (SPEC_BMSTP_DRAFT.md 3.1) when
    its own header lacks `j_m` -- the one column the old query never
    carried. No stamp, no version: the file's own header is the only
    thing consulted."""
    return not os.path.exists(dest_path) or "j_m" not in _header_line(dest_path).split(",")


def build(config, regions=None):
    """Writes `sky/download/twomass_counts/counts_twomass_hpx512__<Region>.csv`
    for each requested region (default: all thirty): one clean-photometry
    row query per Galactic-latitude strip of the region's box (see
    `region_strip_boxes`), concatenated under one header (see module
    docstring for why this is per-source rows, not per-pixel counts). A
    file already present whose header carries `j_m` is skipped; one
    lacking it (predating the widened query) is refetched."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    dest_dir = f"{config.data_root}/sky/download/twomass_counts"
    os.makedirs(dest_dir, exist_ok=True)
    with progress_module.Stage("sky.download.twomass_counts") as st:
        n_regions = len(regions)
        for i, region in enumerate(regions):
            dest_path = f"{dest_dir}/counts_twomass_hpx512__{region}.csv"
            if not _needs_refetch(dest_path):
                print(f"twomass_counts build: {dest_path} present, skipped")
            else:
                if os.path.exists(dest_path):
                    print(f"twomass_counts build: {region} header lacks j_m, refetching")
                boxes = region_strip_boxes(config, region)
                rows = []
                for l0, l1, b0, b1 in boxes:
                    rows.extend(_strip_rows(config, region, l0, l1, b0, b1))
                with open(dest_path, "w") as f:
                    f.write(CSV_HEADER + "\n")
                    for line in rows:
                        f.write(line + "\n")
                print(f"twomass_counts build: {region} -> {dest_path}: {len(rows)} rows "
                      f"({len(boxes)} strip(s))")
            st.tick(i + 1, n_regions, "regions")
        st.done(dest_dir, regions=n_regions)


if __name__ == "__main__":
    build_module.run(build)
