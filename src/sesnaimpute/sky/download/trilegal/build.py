"""The TRILEGAL v2 synthetic field-star catalogues, one per SESNA region.

TRILEGAL 1.6 (Girardi et al. 2005, A&A 436, 895) is served only by an
anonymous web form at `https://stev.oapd.inaf.it/cgi-bin/trilegal` -- there
is no programmatic URL for a fixed query's exact bytes. The 30 regions'
catalogues (91 per-region ascii files, some regions split into parts by the
form's own per-query row cap) were already retrieved by hand and staged at
`<data_root>/sky/download/trilegal-v2/`. This module copies those 91 files
verbatim, byte-for-byte, to the canonical location below, keeping their
names -- the pattern any web-form-only source with no programmatic URL
uses.

The staged directory also carries a `MANIFEST.json`/`MANIFEST.md` receipt
(per-part SHA-256, retrieval timestamps, service output URLs) and a
`PRODUCT.json` bookkeeping index. Neither is copied: hashes and dates are
not science. What SPEC_PRIORS.md section 1.5 needs to reproduce or check
the population -- each region's pointing, queried area, and the TRILEGAL
form settings the query was submitted with -- is retyped below as a plain
constants block, read once from that manifest's own "regions" and
"query_plan" entries.

Feeds SPEC_PRIORS.md section 1.5 (the synthetic field-star population:
STAR, AGB and PAHC share it).
"""

import os
import shutil

from sesnaimpute.build import run

#: Pre-staged location this module copies from -- there is no reachable
#: download URL for these bytes (see module docstring).
_STAGED_DIR = "/Users/jtaylor/Dropbox/Research/SESNA_Complete/sky/download/trilegal-v2"

#: The one query design shared by all 30 regions (staged MANIFEST.json,
#: "service"/"query_plan" blocks): TRILEGAL 1.6, the 2MASS+Spitzer
#: photometric system, Ks<21 Vega depth, no internal-extinction dimming
#: (the census applies its own measured a(d) instead, SPEC_PRIORS.md 1.4),
#: published TRILEGAL defaults otherwise (Chabrier lognormal IMF; disc,
#: bulge and halo structure as served by the form).
TRILEGAL_VERSION = "1.6"
SERVICE_URL = "https://stev.oapd.inaf.it/cgi-bin/trilegal"
PHOTSYS_FILE = "tab_mag_odfnew/tab_mag_2mass_spitzer.dat"
DEPTH_BAND = "Ks (icm_lim=3)"
MAG_LIM_VEGAMAG = 21.0
INTERNAL_EXTINCTION_KIND = 0
IMF_FILE = "tab_imf/imf_chabrier_lognormal.dat"

#: Design floor: the query area was sized, per region, to reach this many
#: TRILEGAL foreground stars (distance d < the region's cloud distance
#: d_r) -- the near-cloud population STAR/AGB/PAHC's joint density needs.
#: Not every region reaches it (`fg_achieved` below); the shortfall is an
#: accepted, disclosed limit of the region's own background/foreground
#: density ratio, not a query defect.
FOREGROUND_TARGET = 3000

#: Per region, read from the staged acquisition's MANIFEST.json "regions"
#: block: the form's pointing (`gc_l`, `gc_b`, Galactic degrees -- the
#: midpoint of the region's coordinate bounding box), the total queried
#: solid angle (`field`, deg^2, split evenly across `n_parts` sub-queries
#: of identical parameters when a region needed more than one), the total
#: rows retrieved, and the achieved foreground count against
#: `FOREGROUND_TARGET`.
REGION_POINTINGS = {
    "AFGL 490": dict(l_deg=142.12770, b_deg=1.88267, area_deg2=0.7177, n_parts=2, rows=261342, fg_achieved=2943),
    "Aquila": dict(l_deg=27.33967, b_deg=5.67689, area_deg2=0.3984, n_parts=4, rows=598514, fg_achieved=309),
    "Auriga-California": dict(l_deg=161.19599, b_deg=-9.79671, area_deg2=3.6160, n_parts=3, rows=311914, fg_achieved=3001),
    "BD+40o4124": dict(l_deg=78.90317, b_deg=2.78451, area_deg2=0.3974, n_parts=3, rows=436193, fg_achieved=3004),
    "Cepheus Flare": dict(l_deg=107.83937, b_deg=15.55979, area_deg2=7.3410, n_parts=3, rows=409882, fg_achieved=3136),
    "Cepheus OB3": dict(l_deg=110.66209, b_deg=2.09128, area_deg2=0.8947, n_parts=4, rows=561687, fg_achieved=2999),
    "Chameleon": dict(l_deg=300.11695, b_deg=-15.91948, area_deg2=4.9336, n_parts=4, rows=600314, fg_achieved=441),
    "CrA": dict(l_deg=359.56799, b_deg=-17.64536, area_deg2=2.5645, n_parts=4, rows=600720, fg_achieved=132),
    "Cygnus X": dict(l_deg=79.48327, b_deg=0.59913, area_deg2=0.1399, n_parts=3, rows=326557, fg_achieved=3271),
    "GGD4, CB34": dict(l_deg=185.42488, b_deg=-3.85323, area_deg2=0.1992, n_parts=1, rows=44498, fg_achieved=3006),
    "IC 5146": dict(l_deg=93.94245, b_deg=-4.91219, area_deg2=0.6560, n_parts=3, rows=315493, fg_achieved=3054),
    "IRAS 20050+2720": dict(l_deg=65.74647, b_deg=-2.66574, area_deg2=0.3210, n_parts=4, rows=600193, fg_achieved=1099),
    "L988": dict(l_deg=90.40746, b_deg=2.29931, area_deg2=0.6223, n_parts=4, rows=601961, fg_achieved=1332),
    "Lupus": dict(l_deg=339.87130, b_deg=11.77564, area_deg2=1.3602, n_parts=4, rows=599209, fg_achieved=64),
    "Mon OB1": dict(l_deg=202.65165, b_deg=1.58258, area_deg2=1.0029, n_parts=3, rows=323408, fg_achieved=3171),
    "Mon R2": dict(l_deg=213.66607, b_deg=-12.23610, area_deg2=0.7346, n_parts=1, rows=49540, fg_achieved=3091),
    "Musca": dict(l_deg=301.06868, b_deg=-8.73965, area_deg2=1.5313, n_parts=4, rows=599999, fg_achieved=121),
    "NGC 7129": dict(l_deg=105.39579, b_deg=9.88562, area_deg2=0.5595, n_parts=1, rows=69936, fg_achieved=3059),
    "North America Nebula": dict(l_deg=84.92385, b_deg=-0.70353, area_deg2=0.2933, n_parts=4, rows=600087, fg_achieved=1208),
    "Ophiuchus": dict(l_deg=355.26181, b_deg=16.04204, area_deg2=2.1988, n_parts=4, rows=600679, fg_achieved=89),
    "Orion A": dict(l_deg=211.07391, b_deg=-19.60595, area_deg2=4.4761, n_parts=1, rows=135326, fg_achieved=3125),
    "Orion B": dict(l_deg=205.89214, b_deg=-14.24848, area_deg2=4.2627, n_parts=2, rows=214360, fg_achieved=3024),
    "Perseus": dict(l_deg=159.20516, b_deg=-19.34244, area_deg2=11.4068, n_parts=3, rows=338689, fg_achieved=2962),
    "Pipe": dict(l_deg=0.24833, b_deg=5.07751, area_deg2=0.0356, n_parts=4, rows=601521, fg_achieved=2),
    "S131": dict(l_deg=98.85778, b_deg=2.92784, area_deg2=0.4737, n_parts=3, rows=307201, fg_achieved=3056),
    "S140": dict(l_deg=107.50646, b_deg=5.12715, area_deg2=0.8077, n_parts=2, rows=246958, fg_achieved=3132),
    "S171": dict(l_deg=118.59941, b_deg=6.12592, area_deg2=0.3247, n_parts=1, rows=66891, fg_achieved=2942),
    "Scorpius": dict(l_deg=2.17559, b_deg=19.99393, area_deg2=3.7895, n_parts=4, rows=600126, fg_achieved=137),
    "Taurus": dict(l_deg=173.56161, b_deg=-17.77185, area_deg2=18.0116, n_parts=4, rows=596324, fg_achieved=624),
    "Vela D": dict(l_deg=263.74753, b_deg=-0.09065, area_deg2=0.4045, n_parts=4, rows=597286, fg_achieved=1288),
}


def _file_names(region, n_parts):
    """The staged file name(s) for `region`: one bare `<region>.dat` when
    the query fit in a single part, else `<region>_part1.dat ..
    _part<n_parts>.dat`.
    """
    if n_parts == 1:
        return (f"{region}.dat",)
    return tuple(f"{region}_part{i}.dat" for i in range(1, n_parts + 1))


def build(config, regions=None, _limit=None):
    """Copies every staged TRILEGAL v2 catalogue file verbatim from
    `_STAGED_DIR` to `<data_root>/sky/download/trilegal`. `regions`
    restricts which regions' files are copied (default: all 30 in
    `REGION_POINTINGS`); `_limit` is a rehearsal knob capping the number
    of files copied per region.
    """
    dest_dir = f"{config.data_root}/sky/download/trilegal"
    os.makedirs(dest_dir, exist_ok=True)
    names = sorted(REGION_POINTINGS) if regions is None else regions

    n_files = 0
    n_bytes = 0
    for region in names:
        info = REGION_POINTINGS[region]
        files = _file_names(region, info["n_parts"])
        if _limit is not None:
            files = files[:_limit]
        for name in files:
            src_path = f"{_STAGED_DIR}/{name}"
            dest_path = f"{dest_dir}/{name}"
            shutil.copyfile(src_path, dest_path)
            file_bytes = os.path.getsize(dest_path)
            n_files += 1
            n_bytes += file_bytes
            print(f"trilegal build: {src_path} -> {dest_path} ({file_bytes} bytes)")
    print(f"trilegal build: {n_files} files, {n_bytes} bytes total, "
          f"{len(names)} regions")


if __name__ == "__main__":
    run(build)
