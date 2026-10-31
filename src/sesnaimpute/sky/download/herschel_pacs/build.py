"""Herschel PACS blue-channel maps (70 or 100 um) for every SESNA region.

Source of truth is the Herschel Science Archive (HSA) observation table,
queried on its TAP service (Herschel Science Centre, archives.esac.esa.int/
hsa); the maps come from the IRSA Herschel High Level Images mirror
(irsa.ipac.caltech.edu/data/Herschel/HHLI/images, JScanam Level 2.5 and
HPF Level 2 maps, served with byte ranges), and from the HSA itself for
observations the mirror lacks. SPEC_PACS_FARIR_DRAFT.md sections 3 and 11.2.

`build(config, regions=None)`, per region:
  1. one ADQL query for the PACS observations (modes 1 scan map and 31
     parallel) in a box around the region, cached as
     `tap_pacs_obs__<Region>.csv`; the observations whose footprint polygon
     touches one of the region's admitted nside-512 pixels are kept
     (`index.region_observations`, a local polygon test);
  2. for each observation the best map that exists: JScanam L2.5, then
     Unimap L2.5, then the L2/L3 map; the mirror first, the HSA when the
     mirror has none. A combined map that already contains an observation
     (the OBSID cards of its header) serves it and is not taken twice;
  3. the projected bytes are printed BEFORE any fetch, the region stops if
     they exceed 10 GB, and a file already on disk is never fetched again;
  4. the per-region index (`index.write_index`) is rebuilt from the files
     on disk. A region with no PACS coverage gets an index with no rows.
The mirror directory listings and the map headers (first bytes, by range
request) are cached beside the maps, and nothing cached is asked for again.
Maps land at `<herschel_pacs_maps>/<obsid>/<level>/<product>/<file>`.
"""

import glob
import os
import re
import shutil
import tarfile
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from astropy.io import fits

from sesnaimpute import progress as progress_module
from sesnaimpute import regions as regions_module
from sesnaimpute.build import run
from sesnaimpute.sky.download.herschel_pacs import index as index_module

TAP_URL = "https://archives.esac.esa.int/hsa/whsa-tap-server/tap/sync"
HSA_DATA_URL = "https://archives.esac.esa.int/hsa/whsa-tap-server/data"
IRSA_URL = "https://irsa.ipac.caltech.edu/data/Herschel/HHLI/images"
MAX_REGION_BYTES = 10 * 1024 ** 3
# Observations are matched on their centre inside a box this much larger than
# the region, in degrees: the largest PACS map is about 4 deg across.
BOX_MARGIN_DEG = 4.0
# A combined map's centre lies within this of the centre of an observation it
# contains (degrees); only maps this close are header-checked for that.
NEAR_DEG = 1.5
# Bytes per map pixel of the three float32 layers, and the pixel scale of each
# observing mode, for projecting an HSA map the mirror does not list.
BYTES_PER_PIXEL = 12
PIXEL_ARCSEC = {1: 1.6, 31: 3.2}
HEADER_BYTES = 131072
MODE_DIR = {1: "PacsPhoto", 31: "PacsParallel"}
LEVEL_DIR = {"L20": "level2", "L25": "level2_5", "L30": "level3"}
HSA_LEVEL = {"LEVEL2_5_PROCESSED": "LEVEL2_5", "LEVEL3_PROCESSED": "LEVEL3",
             "LEVEL2_PROCESSED": "LEVEL2"}
# the two cameras (final ruling 2026-10-27, spec 12: P160 too): the mirror's band directories, and the
# product names -- blue maps end in B (the archive names the 100 um Level 3 stack G, HPPJSMAPG), red maps in R
CHANNELS = {"blue": (70, 100), "red": (160,)}
PRODUCT = {"blue": re.compile(r"(HPPJSMAP[BG]|HPPUNIMAP[BG]|HPPPMAP[BG]).*\.fits(\.gz)?$"),
           "red": re.compile(r"(HPPJSMAPR|HPPUNIMAPR|HPPPMAPR).*\.fits(\.gz)?$")}
PRODUCT_ORDER = {"blue": ("HPPJSMAPB", "HPPJSMAPG", "HPPUNIMAPB", "HPPUNIMAPG", "HPPPMAPB", "HPPPMAPG"),
                 "red": ("HPPJSMAPR", "HPPUNIMAPR", "HPPPMAPR")}
_LISTING_ROW = re.compile(
    r'href="((\d{4})([pm])(\d{4})_(\d+)_(Pacs\w+?)_L(\d\d)_(HPP\w+?)([BR])_SPG[\d.]+\.fits)">'
    r"[^<]*</a>\s+\S+\s+\S+\s+([\d.]+)([KMG]?)")


# ----- network -------------------------------------------------------------

def _open(url, headers=None, data=None, timeout=600):
    req = urllib.request.Request(url, headers=dict(headers or {}), data=data)
    return urllib.request.urlopen(req, timeout=timeout)


def _atomic_write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as out:
        out.write(data)
    os.replace(path + ".part", path)


def stream_to(url, path, expect_bytes=None):
    """Streams `url` to `path` through `path`.part, resuming by byte range
    from whatever the .part holds (the mirror honours ranges; if a server
    answers 200 the part restarts). The file appears only when complete: its
    length is `expect_bytes` when given and a whole number of FITS blocks
    when it is a FITS file. A file already at `path` is left alone."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"Accept-Encoding": "identity"}
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        resp = _open(url, headers)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and expect_bytes == have:
            resp = None
        else:
            raise
    if resp is not None:
        with resp:
            if resp.status == 200:
                have = 0
            with open(part, "r+b" if have else "wb") as out:
                out.seek(have)
                shutil.copyfileobj(resp, out, 1 << 22)
    size = os.path.getsize(part)
    if expect_bytes is not None and size != expect_bytes:
        raise RuntimeError(f"{path}: {size} of {expect_bytes} bytes; partial kept")
    if path.endswith(".fits") and size % 2880:
        raise RuntimeError(f"{path}: not a whole number of FITS blocks; partial kept")
    os.replace(part, path)
    print(f"fetch: {url} -> {path} ({size} bytes)", flush=True)
    return path


def query_region(config, region):
    """The region's PACS observation table (HSA TAP), cached as CSV and never
    asked for again while present. Returns the CSV path."""
    path = index_module.tap_path(config, region)
    if os.path.exists(path):
        return path
    centres, _ = index_module.region_pixels(config, region)
    dec_lo, dec_hi = centres[:, 1].min() - BOX_MARGIN_DEG, centres[:, 1].max() + BOX_MARGIN_DEG
    cosd = np.cos(np.radians(min(89.0, max(abs(dec_lo), abs(dec_hi)))))
    ra = np.sort(centres[:, 0])
    gap = np.diff(np.r_[ra, ra[0] + 360.0])
    k = int(np.argmax(gap))                   # the largest empty RA stretch is outside the box
    ra_lo, ra_hi = ra[(k + 1) % len(ra)], ra[k]
    wide = BOX_MARGIN_DEG / cosd
    lo, hi = (ra_lo - wide) % 360.0, (ra_hi + wide) % 360.0
    ra_clause = (f"ra BETWEEN {lo:.4f} AND {hi:.4f}" if lo <= hi
                 else f"(ra >= {lo:.4f} OR ra <= {hi:.4f})")
    adql = ("SELECT observation_id, observing_mode_oid, ra, dec, polygon_fov, target_name, "
            "proposal_id, obsstate FROM hsa.v_active_observation "
            f"WHERE instrument_oid = 1 AND observing_mode_oid IN (1, 31) AND {ra_clause} "
            f"AND dec BETWEEN {dec_lo:.4f} AND {dec_hi:.4f}")
    return _tap_csv(adql, path, "observation_id")


def _tap_csv(adql, path, first_column):
    params = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
                                     "MAXREC": 200000, "QUERY": adql}).encode()
    with _open(TAP_URL, data=params) as resp:
        body = resp.read()
    if not body.lstrip().lower().startswith(first_column.encode()):
        raise RuntimeError(f"TAP answer is not the expected CSV: {body[:200]!r}")
    _atomic_write(path, body)
    print(f"tap: {path} ({len(body)} bytes)", flush=True)
    return path


def query_scans(config, region, obsids):
    """Each observation's scan direction (nominal/orthogonal) and spacecraft position angle, from the
    HSA's per-observation metadata, cached as CSV and never asked for again while present. The PSF of a
    parallel-mode map needs both of its scans' angles and the map header carries only one
    (sky.derived.pacs.psf.published_components)."""
    path = index_module.scan_path(config, region)
    if os.path.exists(path) or not obsids:
        return path
    ids = ",".join(f"'{o}'" for o in sorted(obsids))
    adql = ("SELECT o.observation_id, m.scandir, m.posangle, m.scanvelo FROM hsa.v_active_observation o "
            "JOIN hsa.metadata_expert_panels m ON o.observation_oid = m.observation_oid "
            f"WHERE o.instrument_oid = 1 AND m.scandir IS NOT NULL AND o.observation_id IN ({ids})")
    return _tap_csv(adql, path, "observation_id")


def hsa_p70_ids(config, region, obsids):
    """Which of `obsids` carry 70 um data per the HSA point-source table, a
    hint for projecting HSA fetches only (the map header decides the band).
    Cached as CSV; empty when the table cannot be queried."""
    path = f"{index_module.maps_root(config)}/tap_pacs_p70_obs__{index_module.safe_name(region)}.csv"
    if not os.path.exists(path):
        if not obsids:
            return set()
        adql = ("SELECT DISTINCT obsid FROM hsa.pacs_point_source_070 WHERE obsid IN ("
                + ",".join(sorted(obsids)) + ")")
        try:
            _tap_csv(adql, path, "obsid")
        except Exception as exc:
            print(f"tap: 70 um hint unavailable ({exc})")
            return set()
    with open(path) as src:
        return {ln.strip().strip('"') for ln in src.read().splitlines()[1:] if ln.strip()}


# ----- the IRSA mirror -----------------------------------------------------

def _hours(ra, dec):
    """The (hour, sign) directories of the mirror that can hold a map whose
    centre is within about 2.5 deg of (ra, dec)."""
    wide = 2.5 / max(np.cos(np.radians(dec)), 0.05)
    h0 = int(ra // 15) % 24
    hs = {h0}
    if ra % 15 < wide:
        hs.add((h0 - 1) % 24)
    if ra % 15 > 15 - wide:
        hs.add((h0 + 1) % 24)
    signs = {"p" if dec >= 0 else "m"}
    if abs(dec) < 2.5:
        signs |= {"p", "m"}
    return sorted((h, s) for h in hs for s in signs)


def listing(config, hour, sign, mode, band):
    """The mirror's directory listing of one RA-hour/sign/mode/band, cached
    as the HTML the server gave. Returns file dicts with the map-centre
    position decoded from the name and the size in bytes (as listed)."""
    name = f"{hour:02d}{sign}"
    path = f"{index_module.maps_root(config)}/irsa_listing_{name}_{MODE_DIR[mode]}_{band}.html"
    if not os.path.exists(path):
        try:
            with _open(f"{IRSA_URL}/{name}/{MODE_DIR[mode]}/{band}/") as resp:
                _atomic_write(path, resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            _atomic_write(path, b"NOT FOUND")
    out = []
    for m in _LISTING_ROW.finditer(open(path, errors="replace").read()):
        fn, hhmm, sg, ddmm, primary, _, lvl, mapper, cam, num, unit = m.groups()
        ra = (int(hhmm[:2]) + int(hhmm[2:]) / 60.0) * 15.0
        dec = (int(ddmm[:2]) + int(ddmm[2:]) / 60.0) * (1 if sg == "p" else -1)
        out.append(dict(file=fn, primary=primary, level="L" + lvl, mapper=mapper,
                        channel="red" if cam == "R" else "blue",
                        ra=ra, dec=dec, mode=mode, band=f"P{band}", dir=f"{name}/{MODE_DIR[mode]}/{band}",
                        listed=int(float(num) * {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}[unit])))
    return out


def map_header(config, entry):
    """(obsids, total bytes) of a mirror map, from its first bytes by range
    request, cached as `irsa_header_<file>.txt` (line 1 'TOTAL n', then the
    primary header cards)."""
    path = f"{index_module.maps_root(config)}/irsa_header_{entry['file']}.txt"
    if not os.path.exists(path):
        with _open(f"{IRSA_URL}/{entry['dir']}/{entry['file']}",
                   {"Range": f"bytes=0-{HEADER_BYTES - 1}", "Accept-Encoding": "identity"}) as resp:
            body = resp.read()
            cr = resp.headers.get("Content-Range", "")
            total = int(cr.rsplit("/", 1)[1]) if "/" in cr else len(body)
        cards = body.decode("ascii", "replace")
        end = [i for i in range(0, len(cards), 80) if cards[i:i + 8] == "END     "]
        cards = cards[:((end[0] // 2880) + 1) * 2880] if end else cards
        _atomic_write(path, f"TOTAL {total}\n{cards}".encode())
    text = open(path).read()
    total = int(text.split("\n", 1)[0].split()[1])
    head = fits.Header.fromstring(text.split("\n", 1)[1])
    ids = [str(head[k]) for k in head if k.startswith("OBSID") and k[5:].isdigit()]
    return tuple(ids) or (entry["primary"],), total


def _rank(entry):
    """Preference among the maps holding an observation: a JScanam Level 3 stack (the same
    observations combined, so at least as deep everywhere and deeper where they overlap), then the
    JScanam Level 2.5 scan pair, then anything else."""
    if entry["mapper"].startswith("HPPJS"):
        return 0 if entry["level"] == "L30" else 1 if entry["level"] == "L25" else 2
    return 2


# ----- planning a region ---------------------------------------------------

def plan_region(config, region, obs):
    """What serves the region's observations, per camera: a list of maps, each a dict (key, source
    IRSA/HSA, channel, band, bytes, estimated, obsids, relpath, url), and the observations nothing
    serves, per camera. Each observation needs one blue map and one red map."""
    plan, unserved = [], {}
    for channel in CHANNELS:
        p, u = _plan_channel(config, region, obs, channel)
        plan += p
        unserved[channel] = u
    return plan, unserved


def _plan_channel(config, region, obs, channel):
    by_id = {o["obsid"]: o for o in obs}
    files = {}                                   # file name -> mirror entry
    for o in obs:
        for h, s in _hours(o["ra"], o["dec"]):
            for band in CHANNELS[channel]:
                for e in listing(config, h, s, o["mode"], band):
                    files[e["file"]] = e
    cands, info = [], {}
    # An observation with no JScanam L2.5 map of its own may sit inside a
    # combined map named for another; the maps centred near it are header-checked.
    has_js = {e["primary"] for e in files.values() if _rank(e) == 0}
    lonely = [o for o in obs if o["obsid"] not in has_js]
    for fn, e in sorted(files.items()):
        primary = e["primary"] in by_id and by_id[e["primary"]]["mode"] == e["mode"]
        near = any(o["mode"] == e["mode"] and np.hypot((o["ra"] - e["ra"]) * np.cos(np.radians(o["dec"])),
                                                       o["dec"] - e["dec"]) < NEAR_DEG
                   for o in lonely)
        if primary or near:
            ids, total = map_header(config, e)
            info[fn] = (ids, total)
            cands.append((fn, _rank(e), ids))
    hsa = {}
    covered = {o for _, _, ids in cands for o in ids}
    missing = [o for o in by_id if o not in covered]
    p70 = hsa_p70_ids(config, region, set(missing)) if channel == "blue" else set()
    for oid in missing:
        o = by_id[oid]
        px, py = index_module._plane(o["polygon"][:, 0], o["polygon"][:, 1], o["ra"], o["dec"]) \
            if o["polygon"] is not None else (np.array([0.0, 0.5]), np.array([0.0, 0.5]))
        area = (px.max() - px.min()) * (py.max() - py.min()) * 3600.0 ** 2
        est = int(1.1 * BYTES_PER_PIXEL * area / PIXEL_ARCSEC[o["mode"]] ** 2)
        level = HSA_LEVEL.get(o["state"], "LEVEL2")
        band = "P160" if channel == "red" else "P70" if oid in p70 else "P100?"
        hsa[f"HSA:{channel}:{oid}"] = dict(obsid=oid, level=level, est=est, band=band)
        cands.append((f"HSA:{channel}:{oid}", {"LEVEL3": 0, "LEVEL2_5": 1}.get(level, 2), (oid,)))
    chosen, unserved = index_module.choose(set(by_id), cands)
    plan = []
    for key in chosen:
        if key in hsa:
            h = hsa[key]
            plan.append(dict(key=key, source="HSA", channel=channel, band=h["band"], bytes=h["est"], estimated=True,
                             obsids=(h["obsid"],), relpath=None, level=h["level"]))
        else:
            e = files[key]
            ids, total = info[key]
            rel = f"{e['primary']}/{LEVEL_DIR[e['level']]}/{e['file'].split('_')[4]}/{e['file']}"
            plan.append(dict(key=key, source="IRSA", channel=channel, band=e["band"], bytes=total, estimated=False,
                             obsids=ids, relpath=rel, url=f"{IRSA_URL}/{e['dir']}/{e['file']}"))
    return plan, unserved


# ----- fetching ------------------------------------------------------------

def fetch_hsa(config, item):
    """One observation's blue maps from the HSA observation tar, the level the
    plan names; the blue-channel map files are extracted to
    `<obsid>/<level>/<product>/`, the rest of the tar is discarded with the tar
    (files this call just fetched)."""
    root = index_module.maps_root(config)
    oid = item["obsids"][0]
    tar_path = f"{root}/{oid}/{oid}_{item['level']}.tar"
    stream_to(HSA_DATA_URL + "?" + urllib.parse.urlencode(
        {"retrieval_type": "OBSERVATION", "observation_id": oid, "instrument_name": "PACS",
         "product_level": item["level"]}), tar_path)
    with tarfile.open(tar_path) as tar:
        for m in tar:
            parts = m.name.split("/")
            match = PRODUCT[item["channel"]].search(parts[-1])
            if not m.isfile() or ".." in parts or m.name.startswith("/") or not match:
                continue
            product = match.group(1)
            dest = f"{root}/{oid}/{LEVEL_DIR[{'LEVEL2': 'L20', 'LEVEL2_5': 'L25', 'LEVEL3': 'L30'}[item['level']]]}/{product}/{parts[-1]}"
            if os.path.exists(dest.removesuffix(".gz")):
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with tar.extractfile(m) as src, open(dest + ".part", "wb") as out:
                shutil.copyfileobj(src, out, 1 << 22)
            os.replace(dest + ".part", dest)
            if dest.endswith(".gz"):
                import gzip
                with gzip.open(dest, "rb") as src, open(dest[:-3] + ".part", "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 22)
                os.replace(dest[:-3] + ".part", dest[:-3])
                os.remove(dest)                    # the compressed copy this call just wrote
    os.remove(tar_path)
    # keep one product: JScanam over Unimap over photProject (spec 11.2); the others are files this
    # call just fetched
    level_dir = f"{root}/{oid}/{LEVEL_DIR[{'LEVEL2': 'L20', 'LEVEL2_5': 'L25', 'LEVEL3': 'L30'}[item['level']]]}"
    present = [p for p in PRODUCT_ORDER[item["channel"]] if glob.glob(f"{level_dir}/{p}/*.fits")]
    for p in present[1:]:
        shutil.rmtree(f"{level_dir}/{p}")
    # the observations the kept map contains (a Level 3 stack holds many): they need no fetch of their own
    kept = glob.glob(f"{level_dir}/{present[0]}/*.fits") if present else []
    return set(index_module.inspect_map(kept[0], root).obsids) if kept else {oid}


def projection(config, region):
    """The region's plan and what it would still fetch, printed per band; nothing is fetched. Returns
    (observations, plan, unserved, summary, bytes still to fetch)."""
    root = index_module.maps_root(config)
    query_region(config, region)
    obs = index_module.region_observations(config, region)
    query_scans(config, region, {o["obsid"] for o in obs})
    plan, unserved = plan_region(config, region, obs) if obs else ([], {})
    gb = {"P70": 0.0, "P100": 0.0, "P100?": 0.0, "P160": 0.0}
    todo = {"P70": 0, "P100": 0, "P100?": 0, "P160": 0}
    for p in plan:
        gb[p["band"]] += p["bytes"] / 1e9
        if not (p["relpath"] and os.path.exists(f"{root}/{p['relpath']}")) and not (
                p["source"] == "HSA" and _hsa_on_disk(root, p)):
            todo[p["band"]] += p["bytes"]
    total = sum(todo.values())
    summary = dict(region=region, n_obs=len(obs), n_maps=len(plan),
                   n_hsa=sum(p["source"] == "HSA" for p in plan), unserved=unserved,
                   gb_p70=gb["P70"], gb_p100=gb["P100"] + gb["P100?"], gb_p160=gb["P160"], gb_to_fetch=total / 1e9)
    print(f"{region}: {len(obs)} observations -> {len(plan)} maps ({summary['n_hsa']} from the HSA); projected "
          f"{gb['P70']:.2f} / {summary['gb_p100']:.2f} / {gb['P160']:.2f} GB at 70 / 100 / 160 um; still to fetch "
          f"{todo['P70'] / 1e9:.2f} / {(todo['P100'] + todo['P100?']) / 1e9:.2f} / {todo['P160'] / 1e9:.2f} GB", flush=True)
    return obs, plan, unserved, summary, total


def project(config, regions=None):
    """Print every region's projection (the volume each band would add) without fetching anything."""
    names = [r.name for r in regions_module.REGIONS] if regions is None else (
        [regions] if isinstance(regions, str) else [getattr(r, "name", r) for r in regions])
    return [projection(config, region)[3] for region in names]


def build(config, regions=None):
    """See the module docstring. `regions` is a list of names (default: all
    in `regions.REGIONS`). Returns one summary dict per region."""
    names = [r.name for r in regions_module.REGIONS] if regions is None else (
        [regions] if isinstance(regions, str) else [getattr(r, "name", r) for r in regions])
    root = index_module.maps_root(config)
    os.makedirs(root, exist_ok=True)
    summaries = []
    # observations already inside a map on disk need no HSA fetch of their own
    served = {"blue": set(), "red": set()}
    for r in index_module.disk_maps(config):
        served["red" if r.band == "P160" else "blue"] |= set(r.obsids)
    with progress_module.Stage("sky.download.herschel_pacs") as st:
        for i, region in enumerate(names):
            obs, plan, unserved, summary, todo = projection(config, region)
            if todo > MAX_REGION_BYTES:
                print(f"{region}: STOP, projection above {MAX_REGION_BYTES / 1e9:.0f} GB; nothing fetched")
            else:
                for p in plan:
                    if p["source"] == "IRSA":
                        try:
                            stream_to(p["url"], f"{root}/{p['relpath']}", p["bytes"])
                        except urllib.error.HTTPError as exc:
                            if not 400 <= exc.code < 500:
                                raise
                            # the mirror refuses or lacks this file (Cygnus X 1342257519, 2026-10-26: 401);
                            # one unavailable map (any 4xx) does not stop the survey
                            print(f"{region}: REFUSED ({exc.code}) {p['url']}", flush=True)
                    elif p["obsids"][0] not in served[p["channel"]] and not _hsa_on_disk(root, p):
                        # one HSA Level 3 tar per observation can hold the same combined map
                        # (Orion A: one stack, nine times); fetch it once
                        try:
                            served[p["channel"]] |= fetch_hsa(config, p)
                        except urllib.error.HTTPError as exc:
                            if not 400 <= exc.code < 500:
                                raise
                            print(f"{region}: REFUSED ({exc.code}) HSA observation {p['obsids'][0]}", flush=True)
            records = index_module.region_index_records(config, region)
            path = index_module.write_index(config, region, records)
            summary["n_indexed"] = len(records)
            summaries.append(summary)
            st.tick(i + 1, len(names), "regions")
        st.done(root, regions=len(names), maps=sum(s["n_maps"] for s in summaries))
    return summaries


def _hsa_on_disk(root, p):
    return any(glob.glob(f"{root}/{p['obsids'][0]}/level*/{prod}/*.fits") for prod in PRODUCT_ORDER[p["channel"]])


if __name__ == "__main__":
    run(build)
