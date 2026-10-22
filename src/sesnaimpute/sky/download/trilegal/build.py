"""The TRILEGAL synthetic field-star catalogues, one per SESNA region.

TRILEGAL (Girardi et al. 2005, A&A 436, 895) is served only by an anonymous
web form at `http://stev.oapd.inaf.it/cgi-bin/trilegal` -- there is no fixed
URL for a query's exact bytes. This module drives that form: it GETs the
form page, reads the version it advertises and every field's current
default value, overrides the pointing, area, photometric system and depth
for the region being built, POSTs the query, and reads the "job running"
reply's own status page and output-file URL: the output file exists (and
grows) well before the job is done, so completion is read from the status
page, not the file's mere existence. Once the status page reads "has
finished", the module fetches the output file, parses its own header
line and saves it as a typed HDF5 table (T6, 2026-10-19), verbatim in
content, at `<data_root>/sky/download/trilegal/<Region>.h5`
(`_part<i>` for a region whose area is split into `n_parts` equal
sub-queries). Each run is a new random realisation of the same population,
so the files on disk are the realisation the current build used, and a
re-run replaces them.

Before T6, every part was saved as ascii text, verbatim; those 931 `.dat`
files already on disk (21 GB) are converted in place, once, by hand, by
the sibling module `sky.download.trilegal.convert` -- never by this
module and never from RUNBOOKtp.sh (the `.dat` files are not deleted; the
owner decides later). `read_ascii_table`, `write_ascii_table_h5` and
`read_region_part_h5` below are the one parse/write/read the fetch and
the converter share, so a reader never again parses TRILEGAL text itself.

Feeds SPEC_PRIORS.md section 1.5 (the synthetic field-star population:
STAR, AGB and PAHC share it).
"""

import html.parser
import io
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

import h5py
import healpy as hp
import numpy as np
import pandas as pd

from sesnaimpute import build as build_module
from sesnaimpute import progress as progress_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute import regions as regions_module
from sesnaimpute.granules import access as granules_access

FORM_URL = "http://stev.oapd.inaf.it/cgi-bin/trilegal"

#: stev.oapd.inaf.it's TLS handshake sends only its leaf certificate, not
#: the intermediate a client needs to build the chain to a trusted root
#: (checked with `openssl s_client`) -- the anonymous form's own
#: server-side gap, not a defect in the query. The two certificates below
#: are the CA's own published chain for that leaf (fetched once from the
#: issuer's AIA URLs named in the leaf certificate: the intermediate from
#: `crt.sectigo.com/ZeroSSLRSADVSSLCA2.crt`, the self-signed root from
#: `crt.sectigo.com/SectigoPublicServerAuthenticationRootR46.p7c`),
#: supplied here so the default trust store can complete verification --
#: nothing about certificate or hostname checking is relaxed.
_ZEROSSL_INTERMEDIATE_PEM = """\
-----BEGIN CERTIFICATE-----
MIIGITCCBAmgAwIBAgIRANCpCCVfjlDl8zltERRwKjcwDQYJKoZIhvcNAQEMBQAw
XzELMAkGA1UEBhMCR0IxGDAWBgNVBAoTD1NlY3RpZ28gTGltaXRlZDE2MDQGA1UE
AxMtU2VjdGlnbyBQdWJsaWMgU2VydmVyIEF1dGhlbnRpY2F0aW9uIFJvb3QgUjQ2
MB4XDTI1MDkyNDAwMDAwMFoXDTM1MDkyMzIzNTk1OVowRjELMAkGA1UEBhMCQVQx
FTATBgNVBAoTDFplcm9TU0wgR21iSDEgMB4GA1UEAxMXWmVyb1NTTCBSU0EgRFYg
U1NMIENBIDIwggGiMA0GCSqGSIb3DQEBAQUAA4IBjwAwggGKAoIBgQCnXX3Qm/G+
ujylvYIkhJ53ZZ7XM03vXY03sCnO9VWjwMsV0qCQMlvhuRiJwrm4M5jgGehxIiCR
1qT0AyL2rHTWJR07vjJzuNmp7uoKu3HKwixBk9QuXD5aliO8/EDbzdZDcG/Hm5pC
mOeusLtds/UE/Iq24nw5WpcgZE5Ly+F/yCYDuEa7hXLJAPM5SecJIblG1OQ3ukMl
HHNBbDxBpGWTyDudd0DTed0NTgCPs1t8RZPhW9Gt/8nDwFX0pQ4eHfPEB8c6eNl2
sRr2Afp1YErakR+53yEX2SXc2Kz0fbTlUc+To0ULGcJiWNyZwj//DTZ+M4xxsT2T
qjQ4Xfvm2EUTymrXDrh1Pm/wkBouu860c6eeQfNlKlccUyHOSeKCtPIWreMvH3Be
Ydeu3DwI8lefn/VUhSB2Bbz7hX3qz3oMmtSTmWhTnobyKlx1L2b/oloaqpy1cBc/
QiLRSOptYGPjZtX0pRrTVKQXeP2rPUk0y5q/40WRpugSlHCX6aceWnsCAwEAAaOC
AW8wggFrMB8GA1UdIwQYMBaAFFZzWGSV+ZIasBIqBGJ5oUAViCFJMB0GA1UdDgQW
BBRLvvp2hCNEBLnOvjFv6fUyBv8MVzAOBgNVHQ8BAf8EBAMCAYYwEgYDVR0TAQH/
BAgwBgEB/wIBADATBgNVHSUEDDAKBggrBgEFBQcDATATBgNVHSAEDDAKMAgGBmeB
DAECATBUBgNVHR8ETTBLMEmgR6BFhkNodHRwOi8vY3JsLnNlY3RpZ28uY29tL1Nl
Y3RpZ29QdWJsaWNTZXJ2ZXJBdXRoZW50aWNhdGlvblJvb3RSNDYuY3JsMIGEBggr
BgEFBQcBAQR4MHYwTwYIKwYBBQUHMAKGQ2h0dHA6Ly9jcnQuc2VjdGlnby5jb20v
U2VjdGlnb1B1YmxpY1NlcnZlckF1dGhlbnRpY2F0aW9uUm9vdFI0Ni5wN2MwIwYI
KwYBBQUHMAGGF2h0dHA6Ly9vY3NwLnNlY3RpZ28uY29tMA0GCSqGSIb3DQEBDAUA
A4ICAQCJ/3v2/vdexHsdVyXL9aCTQE01YXl23866TVM/LgRpRW+kneZXXZxP0hy4
GnvlqUcxTq97B6qPdQcQxQxpGne7CRn0nWauzqieMcJzYl3fDC2Q/ANyPhyrbwCI
zx9EsRrgfjvuJCaUMtlfYpKqBUYiPOCPAN0HdrLD5hU6oV1tvWVsUzTA43skC3uH
wQM5YPIk0NDJFw3NhQPOIOwbq09T+SYSEZvsJ3t4sA4H3gh03RETNaAwTcTNS/+u
1tAeQUZZmKQyLWYLyoxvbISp/MFr9xqhDqrpAurYVNeiLJ5+4/WZPml20yNZjcxV
KKqRYdEurl8rmI2toCnCDWiEcTDvoYGtz60eYIt7VJID4DrCjTxAWTWh2T5ag2pK
ryNJGt8BFGLtjeD774SxAFn5MGYBVvEK4LXmCjVX78pb6/0Dceo71dlQUK68yftb
+yTeyoqjwgk2L8vNSrkj6UTkvvqXSONFuVU7bvC0O/9bi5MXBv7QUivMNxDsTtaT
IZO7PsSCRmLraQM2EPBgbNL9lRSEYi4Hj+NicT/e87pbhv88k0oec9xGcnkcvZpN
sJBz78mkZfeFIIjh02e2y9ke/Fqw1FbdhHtU2myaFnX0sRyGLmI/vzXSwyvT+Kxl
Wx8FV64z2PBdwd0vXRaAMXomGC0M1vQa5fBDn4dimzewsSiphQ==
-----END CERTIFICATE-----
"""

_SECTIGO_ROOT_R46_PEM = """\
-----BEGIN CERTIFICATE-----
MIIFijCCA3KgAwIBAgIQdY39i658BwD6qSWn4cetFDANBgkqhkiG9w0BAQwFADBf
MQswCQYDVQQGEwJHQjEYMBYGA1UEChMPU2VjdGlnbyBMaW1pdGVkMTYwNAYDVQQD
Ey1TZWN0aWdvIFB1YmxpYyBTZXJ2ZXIgQXV0aGVudGljYXRpb24gUm9vdCBSNDYw
HhcNMjEwMzIyMDAwMDAwWhcNNDYwMzIxMjM1OTU5WjBfMQswCQYDVQQGEwJHQjEY
MBYGA1UEChMPU2VjdGlnbyBMaW1pdGVkMTYwNAYDVQQDEy1TZWN0aWdvIFB1Ymxp
YyBTZXJ2ZXIgQXV0aGVudGljYXRpb24gUm9vdCBSNDYwggIiMA0GCSqGSIb3DQEB
AQUAA4ICDwAwggIKAoICAQCTvtU2UnXYASOgHEdCSe5jtrch/cSV1UgrJnwUUxDa
ef0rty2k1Cz66jLdScK5vQ9IPXtamFSvnl0xdE8H/FAh3aTPaE8bEmNtJZlMKpnz
SDBh+oF8HqcIStw+KxwfGExxqjWMrfhu6DtK2eWUAtaJhBOqbchPM8xQljeSM9xf
iOefVNlI8JhD1mb9nxc4Q8UBUQvX4yMPFF1bFOdLvt30yNoDN9HWOaEhUTCDsG3X
ME6WW5HwcCSrv0WBZEMNvSE6Lzzpng3LILVCJ8zab5vuZDCQOc2TZYEhMbUjUDM3
IuM47fgxMMxF/mL50V0yeUKH32rMVhlATc6qu/m1dkmU8Sf4kaWD5QazYw6A3OAS
VYCmO2a0OYctyPDQ0RTp5A1NDvZdV3LFOxxHVp3i1fuBYYzMTYCQNFu31xR13NgE
SJ/AwSiItOkcyqex8Va3e0lMWeUgFaiEAin6OJRpmkkGj80feRQXEgyDet4fsZfu
+Zd4KKTIRJLpfSYFplhym3kT2BFfrsU4YjRosoYwjviQYZ4ybPUHNs2iTG7sijbt
8uaZFURww3y8nDnAtOFr94MlI1fZEoDlSfB1D++N6xybVCi0ITz8fAr/73trdf+L
HaAZBav6+CuBQug4urv7qv094PPK306Xlynt8xhW6aWWrL3DkJiy4Pmi1KZHQ3xt
zwIDAQABo0IwQDAdBgNVHQ4EFgQUVnNYZJX5khqwEioEYnmhQBWIIUkwDgYDVR0P
AQH/BAQDAgGGMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQEMBQADggIBAC9c
mTz8Bl6MlC5w6tIyMY208FHVvArzZJ8HXtXBc2hkeqK5Duj5XYUtqDdFqij0lgVQ
YKlJfp/imTYpE0RHap1VIDzYm/EDMrraQKFz6oOht0SmDpkBm+S8f74TlH7Kph52
gDY9hAaLMyZlbcp+nv4fjFg4exqDsQ+8FxG75gbMY/qB8oFM2gsQa6H61SilzwZA
Fv97fRheORKkU55+MkIQpiGRqRxOF3yEvJ+M0ejf5lG5Nkc/kLnHvALcWxxPDkjB
JYOcCj+esQMzEhonrPcibCTRAUH4WAP+JWgiH5paPHxsnnVI84HxZmduTILA7rpX
DhjvLpr3Etiga+kFpaHpaPi8TD8SHkXoUsCjvxInebnMMTzD9joiFgOgyY9mpFui
TdaBJQbpdqQACj7LzTWb4OE4y2BThihCQRxEV+ioratF4yUQvNs+ZUH7G6aXD+u5
dHn5HrwdVw1Hr8Mvn4dGp+smWg9WY7ViYG4A++MnESLn/pmPNPW56MORcr3Ywx65
LvKRRFHQV80MNNVIIb/bE/FmJUNS0nAiNs2fxBx1IK1jcmMGDw4nztJqDby1ORrp
0XZ60Vzk50lJLVU3aPAaOpg+VBeHVOmmJ1CJeyAvP/+/oYtKR5j/K3tJPsMpRmAY
QqszKbrAKbkTidOIijlBO8n9pu0f9GBj39ItVQGL
-----END CERTIFICATE-----
"""

_SSL_CONTEXT = ssl.create_default_context()
_SSL_CONTEXT.load_verify_locations(cadata=_ZEROSSL_INTERMEDIATE_PEM + _SECTIGO_ROOT_R46_PEM)

#: The query design shared by all 30 regions: the 2MASS+Spitzer
#: photometric system, Ks<21 Vega depth (Ks is the 3rd band the form
#: numbers filters in for that system -- J, H, Ks, ...), no
#: internal-extinction dimming (the census applies its own measured a(d)
#: instead, SPEC_PRIORS.md 1.4), Chabrier lognormal IMF. Everything else
#: the form POST carries is read from the form's own defaults, not stated
#: here.
PHOTSYS_FILE = "tab_mag_odfnew/tab_mag_2mass_spitzer.dat"
DEPTH_ICM_LIM = "3"
DEPTH_MAG_LIM = "21"
INTERNAL_EXTINCTION_KIND = "0"
IMF_FILE = "tab_imf/imf_chabrier_lognormal.dat"

#: The form's own stated per-query area cap ("Total field area", max=10
#: deg2) and CPU-time budget; a region whose queried area needs more than
#: one sub-query is split into `n_parts` equal-area queries of otherwise
#: identical parameters, polled and saved one at a time.
FORM_MAX_AREA_DEG2 = 10.0
POLL_INTERVAL_S = 15
POLL_TIMEOUT_S = 900

#: The pointing grid spacing (SPEC_PRIORS.md section 1.5, "pointings every
#: 1-2 degrees"): a region's admitted footprint (the granules product's own
#: per-source nside-512 pixels) is covered by one TRILEGAL pointing per
#: occupied cell of a grid this wide, anchored on the region's own
#: `REGION_POINTINGS` centre so a region whose whole footprint fits inside
#: one cell keeps that historical single pointing exactly (NGC 7129: 0.5
#: deg across, one pointing, unchanged).
POINTING_GRID_DEG = 1.5

#: Per region: the historical single pointing (`gc_l`, `gc_b`, Galactic
#: degrees -- the midpoint of the region's coordinate bounding box), now
#: the ANCHOR every pointing grid is built around, and the solid angle
#: (`field`, deg^2) EVERY pointing in the region queries -- the area
#: convention (owner ruling 2026-09-06): one pointing simulates the same
#: area a region used to simulate as a whole, so a region's total
#: simulated area is `n_pointings * area_deg2`, not a share of it. Split
#: evenly across `n_parts` sub-queries per pointing when the per-pointing
#: area exceeds the form's own cap.
REGION_POINTINGS = {
    "AFGL 490": dict(l_deg=142.12770, b_deg=1.88267, area_deg2=0.7177, n_parts=2),
    "Aquila": dict(l_deg=27.33967, b_deg=5.67689, area_deg2=0.3984, n_parts=4),
    "Auriga-California": dict(l_deg=161.19599, b_deg=-9.79671, area_deg2=3.6160, n_parts=3),
    "BD+40o4124": dict(l_deg=78.90317, b_deg=2.78451, area_deg2=0.3974, n_parts=3),
    "Cepheus Flare": dict(l_deg=107.83937, b_deg=15.55979, area_deg2=7.3410, n_parts=3),
    "Cepheus OB3": dict(l_deg=110.66209, b_deg=2.09128, area_deg2=0.8947, n_parts=4),
    "Chameleon": dict(l_deg=300.11695, b_deg=-15.91948, area_deg2=4.9336, n_parts=4),
    "CrA": dict(l_deg=359.56799, b_deg=-17.64536, area_deg2=2.5645, n_parts=4),
    "Cygnus X": dict(l_deg=79.48327, b_deg=0.59913, area_deg2=0.1399, n_parts=3),
    "GGD4, CB34": dict(l_deg=185.42488, b_deg=-3.85323, area_deg2=0.1992, n_parts=1),
    "IC 5146": dict(l_deg=93.94245, b_deg=-4.91219, area_deg2=0.6560, n_parts=3),
    "IRAS 20050+2720": dict(l_deg=65.74647, b_deg=-2.66574, area_deg2=0.3210, n_parts=4),
    "L988": dict(l_deg=90.40746, b_deg=2.29931, area_deg2=0.6223, n_parts=4),
    "Lupus": dict(l_deg=339.87130, b_deg=11.77564, area_deg2=1.3602, n_parts=4),
    "Mon OB1": dict(l_deg=202.65165, b_deg=1.58258, area_deg2=1.0029, n_parts=3),
    "Mon R2": dict(l_deg=213.66607, b_deg=-12.23610, area_deg2=0.7346, n_parts=1),
    "Musca": dict(l_deg=301.06868, b_deg=-8.73965, area_deg2=1.5313, n_parts=4),
    "NGC 7129": dict(l_deg=105.39579, b_deg=9.88562, area_deg2=0.5595, n_parts=1),
    "North America Nebula": dict(l_deg=84.92385, b_deg=-0.70353, area_deg2=0.2933, n_parts=4),
    "Ophiuchus": dict(l_deg=355.26181, b_deg=16.04204, area_deg2=2.1988, n_parts=4),
    "Orion A": dict(l_deg=211.07391, b_deg=-19.60595, area_deg2=4.4761, n_parts=1),
    "Orion B": dict(l_deg=205.89214, b_deg=-14.24848, area_deg2=4.2627, n_parts=2),
    "Perseus": dict(l_deg=159.20516, b_deg=-19.34244, area_deg2=11.4068, n_parts=3),
    "Pipe": dict(l_deg=0.24833, b_deg=5.07751, area_deg2=0.0356, n_parts=4),
    "S131": dict(l_deg=98.85778, b_deg=2.92784, area_deg2=0.4737, n_parts=3),
    "S140": dict(l_deg=107.50646, b_deg=5.12715, area_deg2=0.8077, n_parts=2),
    "S171": dict(l_deg=118.59941, b_deg=6.12592, area_deg2=0.3247, n_parts=1),
    "Scorpius": dict(l_deg=2.17559, b_deg=19.99393, area_deg2=3.7895, n_parts=4),
    "Taurus": dict(l_deg=173.56161, b_deg=-17.77185, area_deg2=18.0116, n_parts=4),
    "Vela D": dict(l_deg=263.74753, b_deg=-0.09065, area_deg2=0.4045, n_parts=4),
}


# ---------------------------------------------------------------------------
# T6 (2026-10-19): the ascii<->HDF5 schema shared by the fetch, the
# in-place converter (`sky.download.trilegal.convert`) and every reader.
# ---------------------------------------------------------------------------

#: One attrs_registry stem for every TRILEGAL column this project saves,
#: whichever query produced it: a TRILEGAL column's own meaning (Gc,
#: logAge, ... Mact) is the service's, fixed since Girardi et al. 2005,
#: A&A 436, 895, and is the same column whether it reaches disk from the
#: survey's own 2MASS+Spitzer query (`fetch_region_part`) or the one
#: Gaia+Tycho2+2MASS colour query (`sky.download.trilegal.colour`).
ASCII_STEM = "trilegal_ascii"


def sanitize_column_name(name):
    """A TRILEGAL header token as a valid HDF5 dataset name: the
    brackets TRILEGAL wraps a band name in (`[3.6]`) are dropped, and the
    one character HDF5 reserves as its own path separator, `/` (in
    `[M/H]` and `m2/m1`), is replaced with `_`. The verbatim token is
    recovered from the file's own `HEADER` attr, so sanitising loses no
    information a reader needs."""
    return name.replace("[", "").replace("]", "").replace("/", "_")


#: Every real column is stored float64 (owner ruling 2026-10-19): the
#: readers' products must be bit-identical to what the ascii parse gave,
#: and float32 was measured to move them -- the three atmosphere axes
#: (`logTe`, `logg`, `[M/H]`) and `m-M0` decide a DISCRETE template or
#: colour-table bin in `population.field_stars` and `sky.derived.
#: trilegal_colour` (flips at up to 10 template positions and 22 bin
#: matches on Cygnus X's 6.6M raw stars), and even a magnitude stored
#: float32 moved the field-star fluxes by ~2e-6 relative, above the
#: ~1e-13 rounding bar. The cost is ~4 GB more disk over the 931 files.
def column_dtype(sanitized_name):
    """`int16` for `Gc` (the service's own small integer Galactic-component
    code); float64 for every other column (the comment above)."""
    if sanitized_name == "Gc":
        return np.int16
    return np.float64


def _parse_ascii_table(text):
    """One TRILEGAL ascii reply, as `(header_line, names, df)`: the
    verbatim header line (with its trailing newline), its whitespace-split
    column names, and the parsed table -- dropped of any trailing blank
    row (the service's own reply, module docstring)."""
    header_line = text.split("\n", 1)[0] + "\n"
    names = tuple(header_line.lstrip("#").split())
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", comment="#", header=None,
                      names=names, engine="c")
    df = df.dropna(subset=[names[0]])
    if len(df) == 0:
        raise ValueError("parsed TRILEGAL ascii table has no rows")
    return header_line, names, df


def read_ascii_table(path):
    """`_parse_ascii_table` on one file at `path`, the one parse the
    fetch (on fresh bytes) and `sky.download.trilegal.convert` (on a
    `.dat` already at rest) share."""
    with open(path, "r") as fh:
        text = fh.read()
    try:
        return _parse_ascii_table(text)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def write_ascii_table_h5(dest_path, header_line, names, df, query=None):
    """Writes one TRILEGAL ascii table's columns as typed HDF5 datasets,
    named exactly as the header names them (sanitised,
    `sanitize_column_name`), plus the verbatim header line as a file
    attr -- the one schema `fetch_region_part` (a fresh pull) and
    `sky.download.trilegal.convert` (the 931 files already on disk) both
    write (T6 items 1-2). Download never computes (CODING_RULES_BMSTP.md
    5a): every dataset is one ascii column, cast to its stored dtype,
    nothing derived. `query` (a fresh pull only) is the query that drew
    the file, written as one `QUERY_<FIELD>` attr per entry: provenance,
    read by nothing -- a present file is still never re-queried."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with h5py.File(dest_path, "w") as f:
        f.attrs["HEADER"] = header_line.rstrip("\n")
        for key, value in (query or {}).items():
            f.attrs[f"QUERY_{key.upper()}"] = str(value)
        for name in names:
            clean = sanitize_column_name(name)
            data = df[name].to_numpy(dtype=column_dtype(clean))
            build_module.write_dataset(f, clean, data, *REGISTRY[(ASCII_STEM, clean)],
                                        compression="gzip")


def read_region_part_h5(path):
    """One converted TRILEGAL table (a region/part/pointing file, or the
    one Gaia+2MASS colour query's file) as a DataFrame, columns restored
    to the TRILEGAL header's own spelling (`[M/H]`, `m-M0`, `m2/m1`,
    `[3.6]`, ...) from the file's own `HEADER` attr -- the one function
    every reader of a converted TRILEGAL table uses (T6 item 3):
    `population.field_stars`, `sky.derived.trilegal_colour`, and this
    module's own `convert` identity check. No reader parses TRILEGAL
    ascii text any more."""
    with h5py.File(path, "r") as f:
        header_line = f.attrs["HEADER"]
        names = tuple(header_line.lstrip("#").split())
        data = {name: f[sanitize_column_name(name)][()] for name in names}
    return pd.DataFrame(data)


class _FormFields(html.parser.HTMLParser):
    """Collects, from one TRILEGAL form or reply page: every `<input
    type=text|hidden>`'s value, the checked value of every `<input
    type=radio>` name, the selected `<option>` of every `<select>`, and
    the `<base href>` and `<form action>` targets.
    """

    def __init__(self):
        super().__init__()
        self.fields = {}
        self.base_href = None
        self.form_action = None
        self._select_name = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "input":
            name = a.get("name")
            if name is None:
                return
            kind = a.get("type", "text")
            if kind in ("text", "hidden"):
                self.fields[name] = a.get("value", "")
            elif kind == "radio" and "checked" in a:
                self.fields[name] = a.get("value", "")
        elif tag == "select":
            self._select_name = a.get("name")
        elif tag == "option" and self._select_name is not None:
            if "selected" in a:
                self.fields[self._select_name] = a.get("value", "")
        elif tag == "base":
            self.base_href = a.get("href")
        elif tag == "form" and self.form_action is None:
            self.form_action = a.get("action")

    def handle_endtag(self, tag):
        if tag == "select":
            self._select_name = None


def _get(url, timeout=60):
    """GETs `url`, returns `(text, final_url)` -- `final_url` is where the
    server actually answered, after any redirect (e.g. the form's own
    http->https redirect), the base every relative link on the page
    resolves against.
    """
    with urllib.request.urlopen(url, timeout=timeout, context=_SSL_CONTEXT) as resp:
        text = resp.read().decode("iso-8859-1")
        return text, resp.geturl()


def _multipart_body(fields, boundary):
    parts = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        )
    parts.append(f"--{boundary}--\r\n")
    return "".join(parts).encode("iso-8859-1")


def read_form(form_url=FORM_URL):
    """Reads the live TRILEGAL input form: `(version, action_url, fields)`
    -- the version it advertises (hidden `trilegal_version`), the absolute
    URL its own `<form action>` posts to, and every field's current
    default value.
    """
    text, page_url = _get(form_url)
    parser = _FormFields()
    parser.feed(text)
    base = urllib.parse.urljoin(page_url, parser.base_href) if parser.base_href else page_url
    action_url = urllib.parse.urljoin(base, parser.form_action)
    version = parser.fields.get("trilegal_version")
    return version, action_url, parser.fields


def _post_form(action_url, fields):
    """POSTs `fields` as multipart/form-data to `action_url`, returns the
    reply's `(text, reply_url, hidden_fields)`.
    """
    boundary = "sesnaimpute-trilegal-boundary"
    body = _multipart_body(fields, boundary)
    req = urllib.request.Request(
        action_url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=60, context=_SSL_CONTEXT) as resp:
        text = resp.read().decode("iso-8859-1")
        reply_url = resp.geturl()
    parser = _FormFields()
    parser.feed(text)
    return text, reply_url, parser.fields


#: The service's own refusal page when its job queue is full, and its own
#: instruction on it: "Please wait a few minutes and try again."
BUSY_MARKER = "TRILEGAL is too busy"
BUSY_WAIT_S = 180
BUSY_RETRIES = 20


def _reply_text(html_text):
    """The reply page's visible text, tags stripped, for an error message."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html_text)).strip()


def submit_query(action_url, fields):
    """POSTs one query (`fields`, the form's own defaults with this
    query's overrides) and returns `(output_url, status_fields,
    submitted_at_unix)`: the URL the reply names for the finished
    catalogue (resolved against where the reply itself was served from)
    and the reply's own hidden fields (`outfile`, `outurl`, `submittime`,
    `submitstatus`), which the service asks the client to echo back to
    poll the job's status.
    """
    for attempt in range(BUSY_RETRIES + 1):
        text, reply_url, fields_out = _post_form(action_url, fields)
        outurl = fields_out.get("outurl")
        if outurl:
            submitted_at = time.time()
            return urllib.parse.urljoin(reply_url, outurl), fields_out, submitted_at
        if BUSY_MARKER in text and attempt < BUSY_RETRIES:
            print(f"trilegal submit_query: the server says {BUSY_MARKER!r} -- "
                  f"waiting {BUSY_WAIT_S} s and resubmitting ({attempt + 1}/{BUSY_RETRIES})")
            time.sleep(BUSY_WAIT_S)
            continue
        raise RuntimeError(
            f"trilegal submit_query: no 'outurl' field in the reply from {action_url}: "
            f"{_reply_text(text)[:300]!r}")


def wait_until_finished(action_url, status_fields, timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S):
    """Polls job status by re-posting `status_fields` (the "Refresh this
    page" form) to `action_url` until the reply's heading reads "has
    finished" -- the output file itself exists, and grows, well before
    the job is done, so its mere existence is not the ready signal; the
    status page is.
    """
    fields = dict(status_fields)
    fields["refresh_form"] = "Refresh this page"
    deadline = time.monotonic() + timeout_s
    while True:
        text, _, fields = _post_form(action_url, fields)
        if "has finished" in text:
            return
        if "is running" not in text:
            raise RuntimeError(f"trilegal wait_until_finished: unrecognised reply from {action_url}: "
                                f"{text[:200]!r}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"trilegal wait_until_finished: {action_url} did not finish within {timeout_s}s")
        time.sleep(interval_s)
        fields["refresh_form"] = "Refresh this page"


def fetch_output(output_url):
    """GETs the finished catalogue at `output_url` and returns its bytes."""
    with urllib.request.urlopen(output_url, timeout=120, context=_SSL_CONTEXT) as resp:
        return resp.read()


def _query_fields(defaults, l_deg, b_deg, area_deg2):
    fields = dict(defaults)
    fields.update({
        "gal_coord": "1",
        "gc_l": str(l_deg),
        "gc_b": str(b_deg),
        "field": str(area_deg2),
        "photsys_file": PHOTSYS_FILE,
        "icm_lim": DEPTH_ICM_LIM,
        "mag_lim": DEPTH_MAG_LIM,
        "extinction_kind": INTERNAL_EXTINCTION_KIND,
        "imf_file": IMF_FILE,
        "submit_form": "Submit",
    })
    return fields


#: The submitted form fields a fresh pull records on its own file
#: (`write_ascii_table_h5`'s `query`): the pointing, the area, the
#: service's version and every field this module sets itself.
QUERY_RECORD_FIELDS = ("trilegal_version", "gc_l", "gc_b", "field", "photsys_file",
                       "icm_lim", "mag_lim", "extinction_kind", "imf_file")


def fetch_region_part(l_deg, b_deg, area_deg2, dest_path, defaults, action_url,
                      pointing_idx=None, part_idx=None, n_parts=None):
    """Runs one TRILEGAL query (one region, or one part of a region split
    into `n_parts`) and saves it, typed, at `dest_path` (T6: `.h5`, the
    `write_ascii_table_h5` schema `sky.download.trilegal.convert` also
    writes for the ascii already on disk) -- the reply's own values,
    nothing derived, plus the query itself (`QUERY_RECORD_FIELDS`, and
    the pointing/part indices) as attrs. Returns `(n_rows, wall_time_s)`
    from POST to the file being saved.
    """
    fields = _query_fields(defaults, l_deg, b_deg, area_deg2)
    output_url, status_fields, submitted_at = submit_query(action_url, fields)
    wait_until_finished(action_url, status_fields)
    data = fetch_output(output_url)
    wall_time_s = time.time() - submitted_at
    header_line, names, df = _parse_ascii_table(data.decode("iso-8859-1"))
    query = {k: fields.get(k, "") for k in QUERY_RECORD_FIELDS}
    query.update(pointing_index=pointing_idx, part_index=part_idx, n_parts=n_parts)
    write_ascii_table_h5(dest_path, header_line, names, df, query=query)
    return len(df), wall_time_s


def region_pointings(config, region):
    """The region's own pointing grid (module docstring, `POINTING_GRID_DEG`):
    one dict per occupied grid cell (`l_deg`, `b_deg`, `area_deg2`,
    `n_parts`), sorted by `(l_deg, b_deg)` for a deterministic pointing
    index. The grid is anchored on `REGION_POINTINGS[region]`'s own centre
    and stepped by `POINTING_GRID_DEG` along each axis until it covers
    every admitted nside-512 pixel the granules product has for this
    region (`granules.access.region_slice`, galactic nested lon/lat); a
    pixel is assigned to its NEAREST node on each axis independently, so a
    footprint entirely inside one cell -- NGC 7129 -- returns exactly the
    one historical pointing, unchanged.
    """
    info = REGION_POINTINGS[region]
    c_l, c_b = info["l_deg"], info["b_deg"]
    rs = granules_access.region_slice(config, region)
    pix512 = np.unique(np.asarray(rs["hpx_pix_512"], dtype=np.int64))
    l, b = hp.pix2ang(512, pix512, nest=True, lonlat=True)
    l = np.where(l > 180.0, l - 360.0, l)

    def axis_nodes(x, c):
        n_below = int(np.ceil(max(0.0, c - float(x.min())) / POINTING_GRID_DEG))
        n_above = int(np.ceil(max(0.0, float(x.max()) - c) / POINTING_GRID_DEG))
        return np.array([c + i * POINTING_GRID_DEG for i in range(-n_below, n_above + 1)])

    l_nodes = axis_nodes(l, c_l)
    b_nodes = axis_nodes(b, c_b)
    il = np.argmin(np.abs(l[:, None] - l_nodes[None, :]), axis=1)
    ib = np.argmin(np.abs(b[:, None] - b_nodes[None, :]), axis=1)
    cells = np.unique(np.stack([il, ib], axis=1), axis=0)
    order = sorted(range(len(cells)),
                    key=lambda k: (l_nodes[cells[k, 0]], b_nodes[cells[k, 1]]))
    return tuple(
        dict(l_deg=float(l_nodes[cells[k, 0]]), b_deg=float(b_nodes[cells[k, 1]]),
             area_deg2=info["area_deg2"], n_parts=info["n_parts"])
        for k in order)


def _file_names(region, n_pointings, pointing_idx, n_parts):
    """The file name(s) one pointing's query is saved under (T6: `.h5`,
    typed -- the historical ascii name with its extension swapped). A
    region with a single pointing keeps the historical bare `<region>.h5`
    (or `<region>_part<i>.h5`) name so a region whose footprint needs no
    more than one cell (NGC 7129) reads its existing cache unchanged; a
    region with more than one pointing gets
    `<region>_pt<pointing_idx>[_part<i>].h5`.
    """
    prefix = region if n_pointings == 1 else f"{region}_pt{pointing_idx}"
    if n_parts == 1:
        return (f"{prefix}.h5",)
    return tuple(f"{prefix}_part{i}.h5" for i in range(1, n_parts + 1))


def build(config, regions=None, _limit=None):
    """Drives the TRILEGAL web form once per pointing (once per part, for a
    pointing whose area is split) and saves each reply, typed, at
    `<data_root>/sky/download/trilegal` (T6: `.h5`, `fetch_region_part`).
    `regions` restricts which regions are queried (default: the survey's
    own region list, `regions.REGIONS` -- `REGION_POINTINGS` also carries
    BD+40o4124, which the survey excludes and the granule map does not
    hold); `_limit` is a rehearsal knob capping the number
    of parts queried per pointing. DOES NOT RUN here (T6 ruling): the
    931 `.dat` files already on disk are converted instead, by
    `sky.download.trilegal.convert`, run by hand.
    """
    with progress_module.Stage("sky.download.trilegal") as st:
        # The query form is read from the server only when a part is actually
        # missing (2026-10-22): a rebuild with every part present must never
        # touch the network -- the earlier unconditional read_form() here
        # stalled the whole runbook whenever the Padova server was down.
        form = {}

        def _form():
            if not form:
                version, action_url, defaults = read_form()
                print(f"trilegal build: form version {version!r}, POST target {action_url}")
                form["action_url"], form["defaults"] = action_url, defaults
            return form["defaults"], form["action_url"]

        dest_dir = f"{config.data_root}/sky/download/trilegal"
        names = (sorted(r.name for r in regions_module.REGIONS) if regions is None
                 else regions)

        # Plan every pointing/part's file name(s) first, so the tick below
        # knows the total unit count before touching the network.
        plan = []
        for region in names:
            pointings = region_pointings(config, region)
            n_pointings = len(pointings)
            print(f"trilegal build: {region!r}: {n_pointings} pointing(s) on the "
                  f"{POINTING_GRID_DEG} deg grid")
            # A region is PRESENT when either its full grid set or its single
            # historical pointing's set is on disk -- the same test
            # `population.field_stars` reads by (it falls back to the single
            # pointing and never reads a partial grid) -- and is then skipped
            # whole: only a region with neither set is fetched.
            grid_names = [_file_names(region, n_pointings, p_idx, pt["n_parts"])
                          for p_idx, pt in enumerate(pointings)]
            single_names = _file_names(region, 1, 0, REGION_POINTINGS[region]["n_parts"])
            if (all(os.path.exists(f"{dest_dir}/{n}") for names_ in grid_names for n in names_)
                    or all(os.path.exists(f"{dest_dir}/{n}") for n in single_names)):
                print(f"trilegal build: {region!r} present, skipped "
                      f"(delete its files to draw a new realisation)")
                continue
            for p_idx, pt in enumerate(pointings):
                n_parts = pt["n_parts"]
                area_per_part = pt["area_deg2"] / n_parts
                if area_per_part > FORM_MAX_AREA_DEG2:
                    raise ValueError(
                        f"trilegal build: {region!r} pointing {p_idx} needs area "
                        f"{area_per_part} deg2 per part, above the form's "
                        f"{FORM_MAX_AREA_DEG2} deg2 cap -- raise n_parts")
                file_names = _file_names(region, n_pointings, p_idx, n_parts)
                if _limit is not None:
                    file_names = file_names[:_limit]
                plan.append((region, p_idx, pt, area_per_part, file_names))

        n_total = sum(len(item[4]) for item in plan) or 1
        n_files = 0
        n_rows = 0
        n_done = 0
        for region, p_idx, pt, area_per_part, file_names in plan:
            for part_idx, name in enumerate(file_names, start=1):
                dest_path = f"{dest_dir}/{name}"
                if os.path.exists(dest_path):
                    print(f"trilegal build: {dest_path} present, skipped "
                          f"(delete it to draw a new realisation)")
                else:
                    defaults, action_url = _form()
                    n_out, wall_time_s = fetch_region_part(
                        pt["l_deg"], pt["b_deg"], area_per_part, dest_path, defaults, action_url,
                        pointing_idx=p_idx, part_idx=part_idx, n_parts=pt["n_parts"])
                    n_files += 1
                    n_rows += n_out
                    print(f"trilegal build: {region!r} pointing {p_idx} "
                          f"(l={pt['l_deg']:.4f}, b={pt['b_deg']:.4f}) -> {dest_path} "
                          f"({n_out} rows, {wall_time_s:.0f}s)")
                n_done += 1
                st.tick(n_done, n_total, "files")
        print(f"trilegal build: {n_files} files, {n_rows} rows total, "
              f"{len(names)} regions")
        st.done(dest_dir, regions=len(names), fetched=n_files, rows=n_rows)


if __name__ == "__main__":
    run(build)
