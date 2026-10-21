"""VISTA Hemisphere Survey (VHS) source rows per region (C19 Part B:
the STAR anchor's deep K axis south of UKIDSS GPS's own footprint --
`sky.derived.vhs_counts`, `ukidss_counts`'s own sibling, bins these
rows the same way; `population.anchor_tiles._combine_ks_axis` takes
the deep axis from UKIDSS where UKIDSS covers a pixel, else from VHS).

Download never computes: this module fetches verbatim per-source rows
(Galactic `l`, `b`, `ksAperMag3`, its post-processing error-bits
column, and the merged star/galaxy class) and leaves the nside-512
binning to `sky.derived.vhs_counts`, the exact split
`sky.download.ukidss_gps` / `sky.derived.ukidss_counts` already use for
the UKIDSS anchor extension (`twomass_counts.build.region_strip_boxes`
is reused here verbatim, the same padded Galactic box in the same
Galactic-latitude strips, so this product's occupied-pixel set is the
2MASS/UKIDSS products' own by construction).

Endpoint. The VISTA Science Archive (VSA, operated by WFAU Edinburgh,
the same consortium and the same underlying WSASQL servlet as UKIDSS's
own WSA) serves its freeform-SQL CGI at a different path than WSA's --
`http://vsa.roe.ac.uk:8080/vdfs/WSASQL` (found from the VSA home page's
own "Freeform SQL" link, `vsa.roe.ac.uk:8080/vdfs/VSQL_form.jsp`, whose
`<form>` carries one field WSA's own form does not, a hidden
`archive=VSA` -- without it the servlet's `database=VHSDR7` connects
but `sqlstmt` fails with a bare `NullPointerException`, no SQL-syntax
complaint, this session measured both ways against the live archive
2026-10-21). `database=VHSDR7` is the latest data release the VSA form
itself offers (`VHSv20240731`); `format=CSV` returns the same two-step
HTML-status-page-with-a-results-link shape as WSA's own CGI (W34
section 2b; confirmed again here against VSA). No login or account was
required for any of this -- the archive answered every query, probe
and full pull, anonymously.

Query, per Galactic-latitude strip (`{l0}`/`{l1}`/`{b0}`/`{b1}` from
`twomass_counts.build.region_strip_boxes`, the same boxes UKIDSS's own
pull reuses)::

    SELECT l, b, ksAperMag3, ksppErrBits, mergedClass FROM vhsSource
    WHERE {l clause} AND b BETWEEN {b0} AND {b1}
      AND ksAperMag3 > 0 AND ksAperMag3 < 17.5
      AND mergedClass IN (-1, -2, -3) AND priOrSec <= 0

The same query shape C19 Part A gives the UKIDSS pull (never the old,
narrow error-bit-and-class cut that C18 found undercounting real
stars): `priOrSec <= 0` keeps one row per source where VIRCAM frame
sets overlap; `ksAperMag3 > 0` drops the archive's own null-value
sentinel (measured -9.999995E8 against the live table, the same
magnitude as WSA's own -999999.5, just carried in `vhsSource`'s own
column precision); `< 17.5` gives `sky.derived.vhs_counts`' own last
half-mag bin edge (17.0) a half-bin of headroom; `mergedClass IN
(-1, -2, -3)` keeps the archive's own star / probable star / probable
galaxy codes, never the unambiguous galaxy ones, with `ksppErrBits`
and `mergedClass` both carried through verbatim so `sky.derived.
vhs_counts`, not this download, selects the point-like classes with no
error-bit cut, exactly as `sky.derived.ukidss_counts` now does.

Footprint (this session, 2026-10-21): a `COUNT(*)` probe query (no
row-returning SELECT, so no results-file round trip) run against each
of this brief's five regions' own padded boxes found VHS data in
Musca's box alone (758967 rows before any further cut) -- NGC 7129,
Perseus, North America Nebula and Vela D all returned zero. This
matches the survey's own design: VHS covers the southern sky at
declination roughly below 0 and explicitly cedes the low Galactic-
latitude strip to VVV (VISTA Variables in the Via Lactea), which is
why Vela D (b near 0, in the plane) returns nothing despite sitting at
a VHS-reachable declination, while Musca (b -7 to -10, off the plane,
far south) is covered. A region outside VHS's footprint still writes
a real, header-only results file -- the coverage record itself, the
same convention `sky.download.ukidss_gps.build` uses.
"""

import os
import re
import time
import urllib.parse
import urllib.request

from sesnaimpute import build as build_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.sky.download.twomass_counts import build as twomass_build

VSA_URL = "http://vsa.roe.ac.uk:8080/vdfs/WSASQL"
#: The one field VSA's own form carries that WSA's does not (module
#: docstring): without it the servlet's `sqlstmt` fails outright.
ARCHIVE = "VSA"
DATABASE = "VHSDR7"
TABLE = "vhsSource"
#: Null-value sentinel floor (module docstring: `vhsSource`'s own
#: -9.999995E8, the same WFCAM/VIRCAM-pipeline sentinel magnitude as
#: WSA's gpsSource, measured against the live table).
K_FLOOR = 0.0
#: Half a bin past `sky.derived.vhs_counts.MAG_EDGES`'s top (17.0), the
#: same headroom `sky.download.ukidss_gps.build.K_CEILING` keeps.
K_CEILING = 17.5
#: C19 Part A's own widened class set (that module's own docstring):
#: star, probable star, probable galaxy -- never the unambiguous
#: galaxy/noise codes. The same query shape as the UKIDSS pull.
MERGED_CLASS_VALUES = (-1, -2, -3)
#: Query timeout requested from the server (seconds) and the local
#: socket timeout around it (server timeout plus transfer headroom).
QUERY_TIMEOUT_S = 250
FETCH_TIMEOUT_S = 300
#: A strip whose query comes back with no results link is halved and
#: retried this many times before the region fails outright; below this
#: width in b, no further halving is attempted (`sky.download.
#: ukidss_gps.build`'s own paging rule).
MAX_SPLIT_DEPTH = 5
MIN_STRIP_DEG = 0.05
CSV_HEADER = "l,b,ksAperMag3,ksppErrBits,mergedClass"
_CSV_LINK_RE = re.compile(r'href="(http://vsa\.roe\.ac\.uk/tmp/tmp_sql/[^"]+\.csv)"')
_ERROR_RE = re.compile(r'<b>(SQL Error:|Error in run\(\):)</b>\s*([^<]*)')


def _l_clause(l0, l1):
    """The `l` WHERE clause for one (l0, l1) span, wrap-safe about the
    l=0/360 seam exactly as `sky.download.ukidss_gps.build._l_clause`
    is (`vhsSource`'s `l` is likewise stored in [0, 360))."""
    if l0 >= 0.0 and l1 <= 360.0:
        return f"l BETWEEN {l0:.6f} AND {l1:.6f}"
    lo, hi = l0 % 360.0, l1 % 360.0
    return f"(l >= {lo:.6f} OR l <= {hi:.6f})"


def strip_query(l0, l1, b0, b1):
    """The exact SQL text for one (l0, l1, b0, b1) query box."""
    return (
        "SELECT l, b, ksAperMag3, ksppErrBits, mergedClass FROM "
        f"{TABLE} WHERE {_l_clause(l0, l1)} AND b BETWEEN {b0:.6f} AND {b1:.6f} "
        f"AND ksAperMag3 > {K_FLOOR:.1f} AND ksAperMag3 < {K_CEILING:.1f} "
        f"AND priOrSec <= 0 AND mergedClass IN "
        f"({', '.join(str(v) for v in MERGED_CLASS_VALUES)})"
    )


def _vsa_query_page(sql):
    """POSTs `sql` to the VSA freeform-SQL CGI and returns the HTML
    status page it answers with (never the data itself -- see module
    docstring). `archive=VSA` is the one field the WSA sibling's own
    POST does not carry."""
    data = urllib.parse.urlencode({
        "archive": ARCHIVE, "disp": "", "formaction": "freeform", "database": DATABASE,
        "sqlstmt": sql, "format": "CSV", "compress": "NONE", "rows": "5",
        "timeout": str(QUERY_TIMEOUT_S),
    }).encode("utf-8")
    with urllib.request.urlopen(VSA_URL, data=data, timeout=QUERY_TIMEOUT_S + 60) as resp:
        return resp.read().decode("iso-8859-1")


def _fetch_url(url, timeout=FETCH_TIMEOUT_S):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("iso-8859-1")


def _service_message(html):
    """A one-line description of what VSA sent back instead of a
    results link: its own `SQL Error:` / `Error in run():` box, else
    the page's first non-empty line."""
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
    in `b` and retried, the archive's own message printed each time;
    below `MIN_STRIP_DEG` or `MAX_SPLIT_DEPTH` splits the region fails
    with that message quoted (`sky.download.ukidss_gps.build`'s own
    paging rule, lifted verbatim)."""
    html = _vsa_query_page(strip_query(l0, l1, b0, b1))
    m = _CSV_LINK_RE.search(html)
    if m is not None:
        return _clean_data_lines(_fetch_url(m.group(1)))
    msg = _service_message(html)
    if depth >= MAX_SPLIT_DEPTH or (b1 - b0) <= MIN_STRIP_DEG:
        raise ValueError(
            f"vhs.build: region {region!r} strip l[{l0:.3f},{l1:.3f}] "
            f"b[{b0:.3f},{b1:.3f}] -- VSA returned no results link after "
            f"paging to {b1 - b0:.4f} deg, it said: {msg}")
    print(f"vhs build: {region} strip b[{b0:.3f},{b1:.3f}] -- VSA said: "
          f"{msg} -- paging smaller", flush=True)
    bmid = (b0 + b1) / 2.0
    return (_strip_rows(region, l0, l1, b0, bmid, depth + 1)
            + _strip_rows(region, l0, l1, bmid, b1, depth + 1))


def build(config, regions=None):
    """Writes `sky/download/vhs/sources_vhs_hpx512__<Region>.csv` for
    each requested region (default: all thirty): one query per Galactic-
    latitude strip of the region's box (`twomass_build.region_strip_boxes`,
    reused verbatim), concatenated under one header. A region outside
    VHS's footprint (module docstring: four of this brief's five regions)
    writes a header-only file -- the coverage record itself. Written to
    a temp path and renamed into place only once every strip has
    succeeded, so a run stopped or failed partway through a region never
    leaves a file at the final name for a rerun's skip-if-present check
    to mistake as complete."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    dest_dir = f"{config.data_root}/sky/download/vhs"
    os.makedirs(dest_dir, exist_ok=True)
    with progress_module.Stage("sky.download.vhs") as st:
        n_regions = len(regions)
        for i, region in enumerate(regions):
            dest_path = f"{dest_dir}/sources_vhs_hpx512__{region}.csv"
            if os.path.exists(dest_path):
                print(f"vhs build: {dest_path} present, skipped")
            else:
                t0 = time.time()
                boxes = twomass_build.region_strip_boxes(config, region)
                tmp_path = dest_path + ".partial"
                n_rows = 0
                try:
                    with open(tmp_path, "w") as f:
                        f.write(CSV_HEADER + "\n")
                        for l0, l1, b0, b1 in boxes:
                            for line in _strip_rows(region, l0, l1, b0, b1):
                                f.write(line + "\n")
                                n_rows += 1
                    os.replace(tmp_path, dest_path)
                except BaseException:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    raise
                dt = time.time() - t0
                nbytes = os.path.getsize(dest_path)
                print(f"vhs build: {region} -> {dest_path}: {n_rows} rows, "
                      f"{nbytes} bytes, {dt:.1f} s ({len(boxes)} strip(s))")
            st.tick(i + 1, n_regions, "regions")
        st.done(dest_dir, regions=n_regions)


if __name__ == "__main__":
    build_module.run(build)
