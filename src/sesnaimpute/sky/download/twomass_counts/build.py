"""2MASS PSC anchor source rows per region (SPEC_BMSTP_DRAFT.md section
5.1's anchor row and section 3.1's 2MASS row), fetched as the point
sources 2MASS itself defines: a real detection in Ks, with no exclusion
on any of Ks's own confusion flags.

IRSA's own Explanatory Supplement to the survey (`fp_psc`'s `cc_flg`
field, https://irsa.ipac.caltech.edu/data/2MASS/docs/releases/allsky/doc/sec4_7.html,
"IV.7 Artifact Identification") states the PSC's own construction:
"Sources which are believed to be real objects on the sky, but may have
positional or brightness measurements affected by nearby artifacts are
flagged in the appropriate bands with the `cc_flg`... Detections that
are highly probable artifacts are not included in the final release
Catalogs." The per-value table (same page) confirms this for each of
`cc_flg`'s five non-zero codes: "d" (diffraction spike) is only set on
"sources brighter than the threshold [that] are included in the release
Catalogs, but are marked as contaminated by the spike" -- the fainter,
truly spurious detections along the spike are "culled from the Catalog
during the Catalog Generation process" before any `cc_flg` value is
written; "p" (persistence) likewise: detections above a 0.5 artifact
probability are culled, and only the 0.1-0.5 band "is included in the
Catalog lists, but flagged"; "s" (stripe) and "c" (photometric
confusion) are stated as photometry on a real, retained source being
biased by a neighbour, with no culling step at all; "b" (bandmerge
confusion) is an ambiguity in matching a real source's own band
detections, not a second, spurious source. So no `cc_flg` value that
survives to a delivered `fp_psc` row marks a source that is not a star
-- 2MASS's own pipeline already removed those before delivery -- and
`ARTIFACT_CC_CODES` below is empty, cited to that page, rather than
assumed non-empty from memory. What "clean photometry" (`cc_flg =
'000'`) had been excluding was therefore never an artifact: it was this
same page's confused-but-real population, at a rate this module's
`ph_qual`/`cc_flg` columns let the derive step measure (SPEC_BMSTP_DRAFT.md
5.1, `studies/cygnus_bright_deficit.md` section 5, `studies/star_count_chain.md`
L1).

The real-detection requirement instead falls on `rd_flg`
(https://irsa.ipac.caltech.edu/data/2MASS/docs/releases/allsky/doc/sec2_2a.html#rd_flg),
which the Supplement names as the origin of the default magnitude:
"Rd_flg values of '1', '2' or '3' generally indicate the best quality
detections... Values of '0', '4', '6' and '9' in a band indicate either
non-detections, or generally poor quality photometry." Its own per-code
text is explicit about which of those are non-detections rather than
poor-quality detections: "0" -- "Source is not detected in this band.
The default magnitude is the 95% confidence upper limit..."; "6" -- "...
used for pairs of sources which are detected and resolved in another
band, but are detected and not resolved in this band" (still an upper
limit, "This differs from a rd_flg='0' because... there is a detection
of the source in this band, but it is not consistently resolved");
"9" -- nominally detected but "the default magnitude is null" (already
excluded below by `k_m IS NOT NULL`). `REAL_DETECTION_EXCLUDED_RD_CODES`
below is `('0', '6')`, the two upper-limit, non-detection codes the
Ks character of `rd_flg` must not carry; codes "1"-"4" are the real
detections (aperture or profile-fit) the fix keeps.

Download never computes -- but unlike the Gaia module beside it, this one
cannot produce a per-pixel, per-bin histogram inside the TAP query
itself: IRSA's `fp_psc` TAP service is Oracle-backed and has no HEALPix
function (`ivo_healpix_index` was tried against it directly and IRSA
returned "ORA-00904: IVO_HEALPIX_INDEX: invalid identifier" -- checked
before this module was written; the table also carries no precomputed
HEALPix column, only an HTM-20 index, `htm20`, on an unrelated
tessellation). So this module fetches verbatim per-source rows -- `J`,
`H`, `Ks`, the per-band photometric-quality, confusion and read flags,
and Galactic position -- and the nside-512 binning that pairs it with
the Gaia module's product happens in `sesnaimpute.sky.derived.twomass_counts`,
the one `derive` step in this pair that legitimately computes. `j_m`/
`h_m`/`ph_qual` feed a second, independent view,
`sesnaimpute.sky.derived.twomass_column_scale` (SPEC_BMSTP_DRAFT.md
3.1): the near-infrared colour of the same clean background stars, at
the map's own nside-512 pixels.

Query, per region (the exact text sent, `{l0}`/`{l1}`/`{b0}`/`{b1}` a
padded Galactic bounding box around the region's occupied nside-512
pixels, from the granule map)::

    SELECT glon, glat, j_m, h_m, k_m, ph_qual, cc_flg, rd_flg FROM fp_psc
    WHERE k_m IS NOT NULL AND k_m < 15.5
      AND NOT (rd_flg LIKE '__0' OR rd_flg LIKE '__6')
      AND glon BETWEEN {l0} AND {l1} AND glat BETWEEN {b0} AND {b1}

`k_m < 15.5` matches the old module's `KS_HIST_LIMIT`, the top of the
half-magnitude histogram (9.0-15.5); the `rd_flg LIKE` pair excludes the
two upper-limit, non-detection codes on the Ks character (the SQL/ADQL
`_` wildcard matches exactly one character, so `LIKE '__0'` reads only
the third, Ks position of the three-character flag) -- 2MASS's own
definition of a real Ks point source, applied here rather than
downstream since it is a selection on 2MASS's own external population,
not a SESNA measurement. No `cc_flg` clause: per the module docstring
above, no code in it marks a non-star. `glon`/`glat` are fp_psc's
native Galactic coordinates -- no frame conversion, so a derive step's
`healpy.ang2pix(512, glon, glat, nest=True, lonlat=True)` lands on this
project's own Galactic nside-512 grid directly. `j_m`/`h_m`/`ph_qual`/
`cc_flg`/`rd_flg` are carried verbatim alongside `k_m`, unfiltered here
except as the `WHERE` above states, and selected on downstream.

A file already on disk whose header lacks `rd_flg` predates this widened
query and is refetched; a file whose header already carries it is
skipped, as before (the same rule the sibling UKIDSS module used for a
newly added column).
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
#: `cc_flg` codes that mark a source as an artifact rather than a real
#: star -- empty. IRSA's Explanatory Supplement (sec4_7.html, "IV.7
#: Artifact Identification") states every non-zero `cc_flg` code
#: remaining in a delivered `fp_psc` row is "believed to be a real
#: object on the sky": "d" and "p" are only assigned to the brighter,
#: retained tail of a probabilistic culling that already removed the
#: spurious detections at that same page's own thresholds, and "s"/"c"/
#: "b" carry no culling step at all -- see the module docstring for the
#: quoted per-code text. Kept as an explicit, cited empty tuple rather
#: than silently dropping the concept.
ARTIFACT_CC_CODES = ()
#: `rd_flg` codes that are not a real Ks detection -- IRSA's Explanatory
#: Supplement (sec2_2a.html#rd_flg): "0" -- "Source is not detected in
#: this band. The default magnitude is the 95% confidence upper limit
#: ..."; "6" -- the same upper-limit construction, "used for pairs of
#: sources which are detected and resolved in another band, but are
#: detected and not resolved in this band." Codes "1"-"4" (aperture or
#: profile-fit magnitudes) are real detections and are kept; "9"
#: (nominally detected, no useful brightness) already yields a null
#: `k_m`, excluded by the query's own `k_m IS NOT NULL`.
REAL_DETECTION_EXCLUDED_RD_CODES = ("0", "6")
BOX_PAD_DEG = 0.1  # more than half an hpx512 pixel's ~6.9' diagonal
#: Every query's Galactic-latitude span is cut to at most this many
#: degrees, so no single sync query -- even the densest region's, whose
#: full-box query timed out server-side -- is large enough for IRSA's
#: synchronous query limit to reject or truncate it.
BOX_STRIP_DEG = 1.0
CSV_HEADER = "glon,glat,j_m,h_m,k_m,ph_qual,cc_flg,rd_flg"


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
    """The exact ADQL text for one (l0, l1, b0, b1) query box: 2MASS's
    own real Ks point source (module docstring) -- no `cc_flg` clause
    (`ARTIFACT_CC_CODES` is empty), and `rd_flg`'s Ks (third) character
    excluded from the two non-detection codes via the SQL `_` single-
    character wildcard."""
    rd_exclusion = " OR ".join(f"rd_flg LIKE '__{c}'" for c in REAL_DETECTION_EXCLUDED_RD_CODES)
    return (
        "SELECT glon, glat, j_m, h_m, k_m, ph_qual, cc_flg, rd_flg FROM "
        f"{TABLE} WHERE k_m IS NOT NULL AND k_m < {KS_HIST_LIMIT:.1f} "
        f"AND NOT ({rd_exclusion}) "
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
    """A file predates this widened query (SPEC_BMSTP_DRAFT.md 5.1's
    anchor row) when its own header lacks `rd_flg` -- the one column no
    earlier query carried. No stamp, no version: the file's own header
    is the only thing consulted (the same rule the sibling UKIDSS
    module used for its own newly added column)."""
    return not os.path.exists(dest_path) or "rd_flg" not in _header_line(dest_path).split(",")


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
                    print(f"twomass_counts build: {region} header lacks rd_flg, refetching")
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
