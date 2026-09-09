"""The Gaia archive's own Gaia DR3 x 2MASS PSC crossmatch, grouped per
nside-512 pixel x 1-mag `G` bin x half-magnitude `Ks` bin (SPEC_BMSTP_
DRAFT.md section 5.1, "joint and marginal bins" row: "the joint (G, Ks)
bin where the Gaia archive's own 2MASS crossmatch gives both magnitudes,
counted per nside-512 pixel over the whole pixel like the two marginals
... G < 19, Ks < 14.3"). This replaces `population.anchor_observed`'s
earlier joint histogram, built from SESNA's own sources with a measured
Ks and a Gaia match: that set lies only on the covered part of its pixel
while the anchored prediction spans the whole pixel, so it deflated
every joint cell by the pixel's own coverage before the atlas applied
that coverage again, and calibrated the model to 0.30-0.68 of the sky's
own 2MASS count (`studies/star_count_chain.md`). The archive's own
crossmatch is complete on the whole pixel on both sides, like the two
marginal counts already are (`sky.download.gaia_counts`,
`sky.download.twomass_counts`).

Download never computes: the per-pixel, per-G-bin, per-Ks-bin counts are
the TAP service's own GROUP BY, the same pattern `sky.download.
gaia_counts` uses, one query per region, the returned rows written
verbatim. Query, per region (`{l0}`/`{l1}`/`{b0}`/`{b1}` the same padded
Galactic bounding box `gaia_counts`/`twomass_counts` build from the
granule map -- `sky.download.twomass_counts`'s own wrap-safe
`_lb_bounding_box`, reused here since this query, like that one, can run
long enough to need the latitude-strip fallback below), sent to ESA's
Gaia archive TAP sync endpoint (aliases lower-case, `gaia_counts`'s own
docstring: ESA's TAP layer quotes a mixed-case `SELECT` alias but not
the matching `GROUP BY` reference)::

    SELECT ivo_healpix_index(9, g.l, g.b) AS hpx_pix_512,
           FLOOR(g.phot_g_mean_mag) AS g_bin_lo,
           FLOOR(2 * t.ks_m) / 2 AS ks_bin_lo,
           COUNT(*) AS n
    FROM gaiadr3.gaia_source AS g
    JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS x ON g.source_id = x.source_id
    JOIN gaiadr1.tmass_original_valid AS t ON t.designation = x.original_ext_source_id
    WHERE x.number_of_neighbours = 1
      AND (t.ph_qual LIKE '__A' OR t.ph_qual LIKE '__B'
           OR t.ph_qual LIKE '__C' OR t.ph_qual LIKE '__D')
      AND g.phot_g_mean_mag IS NOT NULL AND g.phot_g_mean_mag < 19.0
      AND t.ks_m IS NOT NULL AND t.ks_m < 14.3
      AND g.l BETWEEN {l0} AND {l1} AND g.b BETWEEN {b0} AND {b1}
    GROUP BY hpx_pix_512, g_bin_lo, ks_bin_lo

The join, verified from this machine (`TAP_SCHEMA.columns`, both tables,
before this query was written): `gaiadr3.tmass_psc_xsc_best_neighbour`
is the archive's own probabilistic best-match table from a Gaia DR3
source to its nearest 2MASS PSC/XSC counterpart, carrying `source_id`,
`original_ext_source_id`, `number_of_neighbours` and `angular_distance`
(no `original_ext_source_id`-adjacent quality column); a 0.2 deg box in
Lupus returns matched rows in seconds. `gaiadr1.tmass_original_valid` is
the archive's own verbatim copy of the 2MASS PSC, joined by its
`designation` (`x.original_ext_source_id` is that same string) --
carrying `ks_m` and `ph_qual`, but NO `cc_flg` column at all (checked):
`sky.download.twomass_counts`'s own clean cut, `cc_flg = '000'`, has no
counterpart on this table.

Clean-match cut, both fields `tap_schema.columns` documents on the best-
neighbour table (state the cut, module docstring): `number_of_neighbours
= 1`, the archive's own count of 2MASS candidates within its own match
radius -- 1 is an unambiguous match, no competing candidate for the
same Gaia source (measured on this machine, a dense Cygnus X test field:
99.9% of matches carry `number_of_neighbours = 1`, the rest 2).
`angular_distance` is not separately thresholded: within `number_of_
neighbours = 1` matches it already runs 0.002-1.8 arcsec (mean 0.2
arcsec, same test field) -- the archive's own best-neighbour choice, not
a raw nearest-neighbour search a distance cut would need to police.

2MASS's own quality flag on this table is `ph_qual`, not `cc_flg`: since
`cc_flg` does not exist here, the nearest available proxy is the Ks-band
character (third) of `ph_qual` in {A, B, C, D} -- the archive's own
"valid photometric measurement" categories (`tap_schema.columns`'
description of `ph_qual`), excluding X (no valid brightness estimate;
already excluded by `ks_m IS NOT NULL`), U (an upper limit, not a
detection), and F/E (uncertainty could not be determined / a poor
profile-fit). ADQL on this service carries no `SUBSTRING`, so the cut is
written as four `LIKE` patterns on the fixed-width three-character
string. This is a signal-to-noise quality cut, not `cc_flg`'s
confusion/contamination cut -- a disclosed mismatch in what "clean"
selects on the two sides of the joint table, not a like-for-like
substitute (`briefs/reports/W45.md`).

If the whole-region box query is throttled (an error, not silence --
CODING_RULES.md rule 6: an HTTP failure or a response that is not the
expected CSV), the same box is retried as `sky.download.twomass_counts`'s
own Galactic-latitude strips (`region_strip_boxes`), one grouped query
per strip. The per-strip rows are concatenated verbatim under one
header, unsummed: latitude strips do not overlap, so a repeated (pixel,
G bin, Ks bin) key across two strips is two disjoint sources, not a
duplicate, and `sky.derived.gaia_twomass_counts`'s own `np.add.at`
combines them exactly the way `sky.derived.gaia_counts`/`twomass_counts`
already combine a download's own repeated rows.
"""

import os
import time
import urllib.error
import urllib.parse
import urllib.request

from sesnaimpute import build as build_module
from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.sky.download.twomass_counts import build as twomass_build

GAIA_TAP_SYNC_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
GAIA_TABLE = "gaiadr3.gaia_source"
XMATCH_TABLE = "gaiadr3.tmass_psc_xsc_best_neighbour"
TMASS_TABLE = "gaiadr1.tmass_original_valid"
G_LIMIT = 19.0  # SPEC_BMSTP_DRAFT.md 5.1 "joint and marginal bins": G < 19
KS_LIMIT = 14.3  # ... Ks < 14.3, sky.derived.twomass_counts.MAG_EDGES[-1]
#: the archive's own "unambiguous match" cut (module docstring).
NUMBER_OF_NEIGHBOURS_MAX = 1
#: the nearest available proxy to the marginal's `cc_flg = '000'`, since
#: this table carries no `cc_flg` (module docstring): the Ks-band
#: character of `ph_qual` in the archive's own "valid measurement" set.
CLEAN_PH_QUAL_KS_CHARS = ("A", "B", "C", "D")
CSV_HEADER = "hpx_pix_512,g_bin_lo,ks_bin_lo,n"


def _ph_qual_clause():
    return "(" + " OR ".join("t.ph_qual LIKE '__%s'" % c for c in CLEAN_PH_QUAL_KS_CHARS) + ")"


def _l_clause(l0, l1):
    """`g.l` WHERE clause for one (l0, l1) span, seam-safe the same way
    `sky.download.twomass_counts._glon_clause` is for `glon` -- this
    table's own Galactic-longitude column, `l`, not `glon`."""
    if l0 >= 0.0 and l1 <= 360.0:
        return f"g.l BETWEEN {l0:.6f} AND {l1:.6f}"
    lo, hi = l0 % 360.0, l1 % 360.0
    return f"(g.l >= {lo:.6f} OR g.l <= {hi:.6f})"


def box_query(l0, l1, b0, b1):
    """The exact ADQL text for one (l0, l1, b0, b1) Galactic box."""
    return (
        "SELECT ivo_healpix_index(9, g.l, g.b) AS hpx_pix_512, "
        "FLOOR(g.phot_g_mean_mag) AS g_bin_lo, FLOOR(2*t.ks_m)/2 AS ks_bin_lo, "
        "COUNT(*) AS n "
        f"FROM {GAIA_TABLE} AS g "
        f"JOIN {XMATCH_TABLE} AS x ON g.source_id = x.source_id "
        f"JOIN {TMASS_TABLE} AS t ON t.designation = x.original_ext_source_id "
        f"WHERE x.number_of_neighbours = {NUMBER_OF_NEIGHBOURS_MAX} "
        f"AND {_ph_qual_clause()} "
        f"AND g.phot_g_mean_mag IS NOT NULL AND g.phot_g_mean_mag < {G_LIMIT:.1f} "
        f"AND t.ks_m IS NOT NULL AND t.ks_m < {KS_LIMIT:.1f} "
        f"AND {_l_clause(l0, l1)} "
        f"AND g.b BETWEEN {b0:.6f} AND {b1:.6f} "
        "GROUP BY hpx_pix_512, g_bin_lo, ks_bin_lo"
    )


def region_query(config, region):
    """The exact ADQL text this module sends for `region`, when the
    region's whole box fits in one query."""
    l0, l1, b0, b1 = twomass_build._lb_bounding_box(config, region)
    return box_query(l0, l1, b0, b1)


def _tap_sync_csv(query, sync_url=GAIA_TAP_SYNC_URL, timeout=600):
    data = urllib.parse.urlencode({"QUERY": query, "FORMAT": "csv", "LANG": "ADQL-2.0"}).encode("utf-8")
    with urllib.request.urlopen(sync_url, data=data, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _region_rows(config, region):
    """The region's own grouped CSV data rows (header stripped): one
    whole-box query where the archive accepts it, else one query per
    Galactic-latitude strip (module docstring). Returns `(rows,
    n_queries)`."""
    l0, l1, b0, b1 = twomass_build._lb_bounding_box(config, region)
    try:
        text = _tap_sync_csv(box_query(l0, l1, b0, b1))
        lines = text.splitlines()
        if lines and lines[0].strip() == CSV_HEADER:
            return lines[1:], 1
        print(
            "sky.download.gaia_twomass_counts: %r whole-box query did not return the "
            "expected header (%s), retrying by Galactic-latitude strip"
            % (region, twomass_build._service_message(text)), flush=True)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(
            "sky.download.gaia_twomass_counts: %r whole-box query failed (%r), retrying "
            "by Galactic-latitude strip" % (region, exc), flush=True)

    boxes = twomass_build.region_strip_boxes(config, region)
    rows = []
    for sl0, sl1, sb0, sb1 in boxes:
        text = _tap_sync_csv(box_query(sl0, sl1, sb0, sb1))
        lines = text.splitlines()
        if not lines or lines[0].strip() != CSV_HEADER:
            raise ValueError(
                "sky.download.gaia_twomass_counts: region %r glat strip [%.3f, %.3f] -- "
                "archive did not return the expected CSV header, it said: %s"
                % (region, sb0, sb1, twomass_build._service_message(text)))
        rows.extend(lines[1:])
    return rows, len(boxes)


def build(config, regions=None):
    """Writes `sky/download/gaia_twomass_counts/joint-counts_gaia-
    twomass_hpx512__<Region>.csv` for each requested region (default:
    all thirty), the archive's own grouped (pixel, G bin, Ks bin) count
    rows, verbatim (module docstring). Skip if present."""
    if regions is None:
        regions = [r.name for r in regions_module.REGIONS]
    dest_dir = f"{config.data_root}/sky/download/gaia_twomass_counts"
    os.makedirs(dest_dir, exist_ok=True)
    with progress_module.Stage("sky.download.gaia_twomass_counts") as st:
        n_regions = len(regions)
        n_fetched = 0
        n_skipped = 0
        for i, region in enumerate(regions):
            dest_path = f"{dest_dir}/joint-counts_gaia-twomass_hpx512__{region}.csv"
            if os.path.exists(dest_path):
                print(f"sky.download.gaia_twomass_counts: {region}: {dest_path} already on disk, skipped", flush=True)
                n_skipped += 1
            else:
                t0 = time.time()
                rows, n_queries = _region_rows(config, region)
                with open(dest_path, "w") as f:
                    f.write(CSV_HEADER + "\n")
                    for line in rows:
                        f.write(line + "\n")
                wall = time.time() - t0
                print(
                    "sky.download.gaia_twomass_counts: %s -> %s: %d rows (%d quer%s), %.1fs"
                    % (region, dest_path, len(rows), n_queries,
                       "y" if n_queries == 1 else "ies", wall), flush=True)
                n_fetched += 1
            st.tick(i + 1, n_regions, "regions")
        st.done(dest_dir, regions=n_regions, fetched=n_fetched, skipped=n_skipped)


if __name__ == "__main__":
    build_module.run(build)
