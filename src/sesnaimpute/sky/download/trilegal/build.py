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
finished", the module fetches the output file and saves it verbatim as
`<data_root>/sky/download/trilegal/<Region>.dat`
(`_part<i>` for a region whose area is split into `n_parts` equal
sub-queries). Each run is a new random realisation of the same population,
so the files on disk are the realisation the current build used, and a
re-run replaces them.

Feeds SPEC_PRIORS.md section 1.5 (the synthetic field-star population:
STAR, AGB and PAHC share it).
"""

import html.parser
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

from sesnaimpute.build import run

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

#: Per region: the form's pointing (`gc_l`, `gc_b`, Galactic degrees -- the
#: midpoint of the region's coordinate bounding box) and the total queried
#: solid angle (`field`, deg^2, split evenly across `n_parts` sub-queries
#: when a region needs more than one).
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


def submit_query(action_url, fields):
    """POSTs one query (`fields`, the form's own defaults with this
    query's overrides) and returns `(output_url, status_fields,
    submitted_at_unix)`: the URL the reply names for the finished
    catalogue (resolved against where the reply itself was served from)
    and the reply's own hidden fields (`outfile`, `outurl`, `submittime`,
    `submitstatus`), which the service asks the client to echo back to
    poll the job's status.
    """
    text, reply_url, fields_out = _post_form(action_url, fields)
    outurl = fields_out.get("outurl")
    if not outurl:
        raise RuntimeError(f"trilegal submit_query: no 'outurl' field in the reply from {action_url}")
    submitted_at = time.time()
    return urllib.parse.urljoin(reply_url, outurl), fields_out, submitted_at


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


def fetch_region_part(l_deg, b_deg, area_deg2, dest_path, defaults, action_url):
    """Runs one TRILEGAL query (one region, or one part of a region split
    into `n_parts`) and saves the reply verbatim at `dest_path`. Returns
    `(n_bytes, wall_time_s)` from POST to the file being saved.
    """
    fields = _query_fields(defaults, l_deg, b_deg, area_deg2)
    output_url, status_fields, submitted_at = submit_query(action_url, fields)
    wait_until_finished(action_url, status_fields)
    data = fetch_output(output_url)
    wall_time_s = time.time() - submitted_at
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as out:
        out.write(data)
    return len(data), wall_time_s


def _file_names(region, n_parts):
    """The file name(s) `region` is saved under: one bare `<region>.dat`
    when the query fits in a single part, else `<region>_part1.dat ..
    _part<n_parts>.dat`.
    """
    if n_parts == 1:
        return (f"{region}.dat",)
    return tuple(f"{region}_part{i}.dat" for i in range(1, n_parts + 1))


def build(config, regions=None, _limit=None):
    """Drives the TRILEGAL web form once per region (once per part, for a
    region whose area is split) and saves each reply verbatim at
    `<data_root>/sky/download/trilegal`. `regions` restricts which
    regions are queried (default: all 30 in `REGION_POINTINGS`); `_limit`
    is a rehearsal knob capping the number of parts queried per region.
    """
    version, action_url, defaults = read_form()
    print(f"trilegal build: form version {version!r}, POST target {action_url}")

    dest_dir = f"{config.data_root}/sky/download/trilegal"
    names = sorted(REGION_POINTINGS) if regions is None else regions

    n_files = 0
    n_bytes = 0
    for region in names:
        info = REGION_POINTINGS[region]
        n_parts = info["n_parts"]
        area_per_part = info["area_deg2"] / n_parts
        if area_per_part > FORM_MAX_AREA_DEG2:
            raise ValueError(f"trilegal build: {region!r} needs area {area_per_part} deg2 per "
                              f"part, above the form's {FORM_MAX_AREA_DEG2} deg2 cap -- raise n_parts")
        file_names = _file_names(region, n_parts)
        if _limit is not None:
            file_names = file_names[:_limit]
        for name in file_names:
            dest_path = f"{dest_dir}/{name}"
            if os.path.exists(dest_path):
                print(f"trilegal build: {dest_path} present, skipped (delete it to draw a new realisation)")
                continue
            n_out, wall_time_s = fetch_region_part(
                info["l_deg"], info["b_deg"], area_per_part, dest_path, defaults, action_url)
            n_files += 1
            n_bytes += n_out
            print(f"trilegal build: {region!r} -> {dest_path} "
                  f"({n_out} bytes, {wall_time_s:.0f}s)")
    print(f"trilegal build: {n_files} files, {n_bytes} bytes total, "
          f"{len(names)} regions")


if __name__ == "__main__":
    run(build)
