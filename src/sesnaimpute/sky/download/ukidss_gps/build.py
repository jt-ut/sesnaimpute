"""UKIDSS Galactic Plane Survey (GPS) source rows per region (SPEC_BMSTP_DRAFT.md
5.1's STAR anchor; W37 extends the anchor's deep K axis past 2MASS's 14.3
cut -- W38 reads the counts this module's `sky.derived.ukidss_counts`
sibling bins from these rows). GPS reaches K approx 18 (5 sigma) over
|b| lesssim 5 deg at l 15-107 deg and 141-230 deg (Lucas et al. 2008,
MNRAS 391, 136); briefs/reports/W34.md probed the archive and measured
the query mechanics this module follows.

Download never computes: this module fetches verbatim per-source rows
(Galactic `l`, `b`, `k_1AperMag3`, its post-processing error-bits column,
and the merged star/galaxy class) and leaves the nside-512 binning to
`sesnaimpute.sky.derived.ukidss_counts`, exactly the split
`sky.download.twomass_counts` / `sky.derived.twomass_counts` already use
for the 2MASS anchor (its `region_strip_boxes` is reused here verbatim:
the same padded Galactic box, in the same Galactic-latitude strips, so
this product's occupied-pixel set is the 2MASS product's by construction,
W37's identity check).

Endpoint (W34 section 1): the WSA freeform-SQL CGI,
`http://wsa.roe.ac.uk:8080/wsa/WSASQL` (plain http -- TLS on the `wsa`
subdomain does not verify), `database=UKIDSSDR11PLUS`, `format=CSV`.
Unlike IRSA's TAP, a `format=CSV` submission returns an HTML status page
whose body carries a link to a server-generated results file (W34
section 2b measured this two-step shape and this session confirmed it
again, including for a zero-row query, which still yields a real,
header-only results file -- the coverage record for the twenty regions
outside the survey footprint).

Query, per Galactic-latitude strip (`{l0}`/`{l1}`/`{b0}`/`{b1}` from
`twomass_counts.build.region_strip_boxes`)::

    SELECT l, b, k_1AperMag3, k_1ppErrBits, mergedClass FROM gpsSource
    WHERE {l clause} AND b BETWEEN {b0} AND {b1}
      AND k_1AperMag3 > 0 AND k_1AperMag3 < 18.5
      AND mergedClass IN (-1, -2, -3) AND priOrSec <= 0

`priOrSec <= 0` keeps one row per source where WFCAM frame sets overlap
(the archive's own primary-detection flag; the secondaries were 7 % of
the rows in the probe's Cygnus X box); `k_1AperMag3 > 0` drops the WFCAM
null-value sentinel (measured -999999.5, W34 section 2b: 14.7% of an
unfiltered test box). C28 (2026-10-21): the 18-56 uJy STAR members sit
at Ks 17-18.3, beyond the old 17.0 anchor edge and its 17.5 pull cut
(C24f/g). `< 18.5` gives the derive step's new last half-mag bin edge
(18.0) a half-bin of headroom, the same buffer `twomass_counts` keeps
against its own 14.3 cut and this module kept against its own old 17.0
edge.

C28: strips are submitted CONCURRENTLY, a pool of `N_CONCURRENT_STRIPS`
at a time (the archive throttles one client, never a per-strip rate),
each result CSV fetched as soon as its own query answers. A strip
refused with a load/limit message still halves and retries exactly as
before (`_strip_rows`); the pool's own width does not change on a
refusal, only the one strip's own split. Rows are written to the
region's file in `twomass_counts.build.region_strip_boxes`' own strip
order (not completion order), so this module's own output keeps the
same row order it always has -- concurrency changes only how fast the
strips arrive, never the file this module writes.

C19 (2026-10-21): the download no longer applies W34's own post-
processing error-bit cut (`k_1ppErrBits < 256`) or its narrow class cut
(`mergedClass IN (-1, -2)`) -- C18's diagnosis found that cut holding
only 0.77-0.79 of the stars SESNA's own 2MASS photometry shows at Ks
12-14, in the North America Nebula and in Perseus alike, which the
anchor fit then read as a real deficit in the sky rather than in the
cut. `mergedClass` now widens to the archive's own `-1` (star), `-2`
(probable star) and `-3` (probable galaxy) -- never the unambiguous
galaxy/noise codes -- and both `k_1ppErrBits` and `mergedClass` are
still carried through verbatim (the SELECT list and CSV header,
unchanged) so `sky.derived.ukidss_counts`, not this download, selects
the point-like classes at derive time, with no error-bit cut at all
(that module's own docstring).

W34 section 4: no per-query row cap was hit and paging was not needed at
this endpoint's scale; this module still splits a strip in half and
retries, quoting the archive's own message, if a strip's query ever
comes back with no results link (a real refusal or a timeout) rather
than assuming the endpoint never throttles.
"""

import concurrent.futures
import os
import re
import time
import urllib.parse
import urllib.request

from sesnaimpute import build as build_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.sky.download.twomass_counts import build as twomass_build

WSA_URL = "http://wsa.roe.ac.uk:8080/wsa/WSASQL"
DATABASE = "UKIDSSDR11PLUS"
TABLE = "gpsSource"
#: Null-value sentinel floor (W34 section 2b: WFCAM's own -999999.5).
K_FLOOR = 0.0
#: Half a bin past `sky.derived.ukidss_counts.MAG_EDGES`'s top (18.0), the
#: same headroom `twomass_counts` keeps against its own 14.3 cut (C28,
#: 2026-10-21: raised from 17.5 -- the 18-56 uJy STAR members sit at Ks
#: 17-18.3, beyond the old 17.0 anchor edge, C24f/g).
K_CEILING = 18.5
#: C28: strips queried at once, a pool of this width (the archive throttles
#: one client, never a per-strip rate -- the halving-on-refusal rule below
#: is unchanged per strip; this is concurrency across DIFFERENT strips'
#: queries only).
N_CONCURRENT_STRIPS = 6
#: C19 (2026-10-21): the WFCAM pipeline post-processing error-bits
#: threshold W34 section 2b measured (`ERR_BITS_MAX = 256`) is retired
#: from the download query -- C18 found it holding only 0.77-0.79 of
#: SESNA's own 2MASS-measured stars at Ks 12-14 -- and the class cut
#: widens from the two stellar-only codes to every one of the archive's
#: own classes except the unambiguous galaxy/noise ones: `-1` star, `-2`
#: probable star, `-3` probable galaxy. Both columns are still pulled
#: through verbatim so `sky.derived.ukidss_counts` can select the
#: point-like classes itself, with no error-bit cut at all.
MERGED_CLASS_VALUES = (-1, -2, -3)
#: Query timeout requested from the server (seconds) and the local
#: socket timeout around it (server timeout plus transfer headroom).
QUERY_TIMEOUT_S = 250
FETCH_TIMEOUT_S = 300
#: A strip whose query comes back with no results link is halved and
#: retried this many times before the region fails outright; below this
#: width in b, no further halving is attempted (W37's own paging rule,
#: since W34 hit no cap to page against).
MAX_SPLIT_DEPTH = 5
MIN_STRIP_DEG = 0.05
CSV_HEADER = "l,b,k_1AperMag3,k_1ppErrBits,mergedClass"
_CSV_LINK_RE = re.compile(r'href="(http://wsa\.roe\.ac\.uk/tmp/tmp_sql/[^"]+\.csv)"')
_ERROR_RE = re.compile(r'<b>(SQL Error:|Error in run\(\):)</b>\s*([^<]*)')


def _l_clause(l0, l1):
    """The `l` WHERE clause for one (l0, l1) span, wrap-safe about the
    l=0/360 seam exactly as `twomass_build._glon_clause` is for `glon`
    (gpsSource's `l` is likewise stored in [0, 360))."""
    if l0 >= 0.0 and l1 <= 360.0:
        return f"l BETWEEN {l0:.6f} AND {l1:.6f}"
    lo, hi = l0 % 360.0, l1 % 360.0
    return f"(l >= {lo:.6f} OR l <= {hi:.6f})"


def strip_query(l0, l1, b0, b1):
    """The exact SQL text for one (l0, l1, b0, b1) query box."""
    return (
        "SELECT l, b, k_1AperMag3, k_1ppErrBits, mergedClass FROM "
        f"{TABLE} WHERE {_l_clause(l0, l1)} AND b BETWEEN {b0:.6f} AND {b1:.6f} "
        f"AND k_1AperMag3 > {K_FLOOR:.1f} AND k_1AperMag3 < {K_CEILING:.1f} "
        f"AND priOrSec <= 0 AND mergedClass IN "
        f"({', '.join(str(v) for v in MERGED_CLASS_VALUES)})"
    )


def _wsa_query_page(sql):
    """POSTs `sql` to the WSA freeform-SQL CGI and returns the HTML status
    page it answers with (never the data itself -- see module docstring)."""
    data = urllib.parse.urlencode({
        "formaction": "freeform", "database": DATABASE, "sqlstmt": sql,
        "format": "CSV", "compress": "NONE", "rows": "5",
        "timeout": str(QUERY_TIMEOUT_S),
    }).encode("utf-8")
    with urllib.request.urlopen(WSA_URL, data=data, timeout=QUERY_TIMEOUT_S + 60) as resp:
        return resp.read().decode("iso-8859-1")


def _fetch_url(url, timeout=FETCH_TIMEOUT_S):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("iso-8859-1")


def _service_message(html):
    """A one-line description of what WSA sent back instead of a results
    link: its own `SQL Error:` / `Error in run():` box, else the page's
    first non-empty line."""
    m = _ERROR_RE.search(html)
    if m:
        return f"{m.group(1)} {m.group(2)}".strip()
    first = html.strip().split("\n", 1)[0].strip()
    return first if first else "<empty response>"


def _clean_data_lines(csv_text):
    """The results file's data rows only (its `#`-prefixed header/query
    echo dropped), each field stripped of the archive's fixed-width
    padding so every region's file carries one clean row shape."""
    out = []
    for line in csv_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(",".join(part.strip() for part in s.split(",")))
    return out


def _strip_rows(region, l0, l1, b0, b1, depth=0):
    """One strip's clean data rows. A query that comes back with no
    results link (a genuine refusal or a server-side timeout) is halved
    in `b` and retried, the archive's own message printed each time, per
    the brief's paging instruction; below `MIN_STRIP_DEG` or
    `MAX_SPLIT_DEPTH` splits the region fails with that message quoted."""
    html = _wsa_query_page(strip_query(l0, l1, b0, b1))
    m = _CSV_LINK_RE.search(html)
    if m is not None:
        return _clean_data_lines(_fetch_url(m.group(1)))
    msg = _service_message(html)
    if depth >= MAX_SPLIT_DEPTH or (b1 - b0) <= MIN_STRIP_DEG:
        raise ValueError(
            f"ukidss_gps.build: region {region!r} strip l[{l0:.3f},{l1:.3f}] "
            f"b[{b0:.3f},{b1:.3f}] -- WSA returned no results link after "
            f"paging to {b1 - b0:.4f} deg, it said: {msg}")
    print(f"ukidss_gps build: {region} strip b[{b0:.3f},{b1:.3f}] -- WSA said: "
          f"{msg} -- paging smaller", flush=True)
    bmid = (b0 + b1) / 2.0
    return (_strip_rows(region, l0, l1, b0, bmid, depth + 1)
            + _strip_rows(region, l0, l1, bmid, b1, depth + 1))


def build(config, regions=None):
    """Writes `sky/download/ukidss_gps/sources_ukidss_hpx512__<Region>.csv`
    for each requested region (default: all thirty): one query per
    Galactic-latitude strip of the region's box (`twomass_build.region_strip_boxes`,
    reused verbatim), concatenated under one header. A region outside GPS's
    footprint (twenty of the thirty, W34 section 3) writes a header-only
    file -- the coverage record itself. Written to a temp path and renamed
    into place only once every strip has succeeded, so a run stopped or
    failed partway through a region never leaves a file at the final name
    for a rerun's skip-if-present check to mistake as complete."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    dest_dir = f"{config.data_root}/sky/download/ukidss_gps"
    os.makedirs(dest_dir, exist_ok=True)
    with progress_module.Stage("sky.download.ukidss_gps") as st:
        n_regions = len(regions)
        for i, region in enumerate(regions):
            dest_path = f"{dest_dir}/sources_ukidss_hpx512__{region}.csv"
            if os.path.exists(dest_path):
                print(f"ukidss_gps build: {dest_path} present, skipped")
            else:
                t0 = time.time()
                boxes = twomass_build.region_strip_boxes(config, region)
                tmp_path = dest_path + ".partial"
                n_rows = 0
                try:
                    with open(tmp_path, "w") as f:
                        f.write(CSV_HEADER + "\n")
                        # C28: the strips are submitted concurrently (a pool
                        # of N_CONCURRENT_STRIPS at a time), but
                        # `ThreadPoolExecutor.map` hands results back in the
                        # SAME order as `boxes` regardless of which strip's
                        # query answers first, so this file's own row order
                        # (strip by strip, in `boxes`' own order) is exactly
                        # what the sequential pull wrote.
                        n_workers = max(1, min(N_CONCURRENT_STRIPS, len(boxes)))
                        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                            for rows in pool.map(lambda box: _strip_rows(region, *box), boxes):
                                for line in rows:
                                    f.write(line + "\n")
                                    n_rows += 1
                    os.replace(tmp_path, dest_path)
                except BaseException:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    raise
                dt = time.time() - t0
                nbytes = os.path.getsize(dest_path)
                print(f"ukidss_gps build: {region} -> {dest_path}: {n_rows} rows, "
                      f"{nbytes} bytes, {dt:.1f} s ({len(boxes)} strip(s))")
            st.tick(i + 1, n_regions, "regions")
        st.done(dest_dir, regions=n_regions)


if __name__ == "__main__":
    build_module.run(build)
