"""Pulls the SWIRE IRAC-band-merged source catalogue, six blank fields
(Lonsdale et al. 2003, PASP 115, 897; Surace et al. 2005 DR2 release), from
IRSA's TAP service. Download never computes: this module selects the
columns SPEC_PRIORS section 5.1's GAL count needs to build the external
four-band colour distribution `eps_s(S | A_s)` -- position, the four IRAC
aperture-2 fluxes and their uncertainties, per-band stellarity, and the
per-band extended-source flag that separates stars from galaxies -- and
writes them verbatim, upstream column names kept, one CSV per field.

Table names were discovered against IRSA's TAP_SCHEMA (the per-field
scan-listed name for the fully band-merged catalogue, as opposed to the
per-band 24/70/160um tables): `chandra_cat_f05` (CDFS), `elaisn1_cat_s05`,
`elaisn2_cat_s05`, `elaiss1_cat_f05`, `lockman_cat_s05`, `xmm_cat_s05`.

The MIPS 70 um catalogue of the same six fields (`<field>_70_cat_*`) is pulled
the same way into `swire70_<field>.csv`, beside the IRAC files and never
rewriting them (the IRAC table has no 70 um column and the 70 um table no IRAC
key; `sky.derived.swire_galaxies` matches them by position).
"""

import os

from sesnaimpute import build as build_module
from sesnaimpute import progress as progress_module
from sesnaimpute.sky.download import _tap

# Field name -> IRSA TAP table for the SWIRE IRAC-band-merged catalogue.
FIELDS = {
    "ELAIS-N1": "elaisn1_cat_s05",
    "ELAIS-N2": "elaisn2_cat_s05",
    "ELAIS-S1": "elaiss1_cat_f05",
    "Lockman": "lockman_cat_s05",
    "XMM-LSS": "xmm_cat_s05",
    "CDFS": "chandra_cat_f05",
}

# cntr (the page cursor) first, then position, the four-band aperture-2
# fluxes with uncertainties, per-band stellarity, and the per-band
# extended-source flag -- the star-galaxy separation SPEC_PRIORS 5.1 needs.
COLUMNS = (
    "cntr", "ra", "dec",
    "flux_ap2_36", "uncf_ap2_36",
    "flux_ap2_45", "uncf_ap2_45",
    "flux_ap2_58", "uncf_ap2_58",
    "flux_ap2_80", "uncf_ap2_80",
    "stell_36", "stell_45", "stell_58", "stell_80",
    "ext_fl_36", "ext_fl_45", "ext_fl_58", "ext_fl_80",
)

# Field name -> IRSA TAP table for the SWIRE MIPS 70 um single-band catalogue
# (the band-merged IRAC tables above carry no 70 um column and the 70 um tables
# carry no IRAC key, so the 70 um rows are their own sibling file per field,
# matched to the IRAC rows by position downstream).
FIELDS_70 = {
    "ELAIS-N1": "elaisn1_70_cat_s05",
    "ELAIS-N2": "elaisn2_70_cat_s05",
    "ELAIS-S1": "elaiss1_70_cat_f05",
    "Lockman": "lockman_70_cat_s05",
    "XMM-LSS": "xmm_70_cat_s05",
    "CDFS": "chandra_70_cat_f05",
}

# cntr (the page cursor, the 70 um table's own source key), position and its
# uncertainty, the point-response-function flux and its uncertainty (mJy), the
# release's own signal-to-noise, and its extraction flag.
COLUMNS_70 = (
    "cntr", "ra", "dec", "unc_ra", "unc_dec",
    "flux_prf", "uncf_prf", "snr", "flag",
)


def build(config, regions=None):
    """Writes `sky/download/swire/swire_<field>.csv` for each of the six
    SWIRE blank fields. `regions` is accepted for the standard `build`
    signature and ignored: the six fields are fixed by the SWIRE release,
    not by SESNA's own region list.
    """
    dest_dir = f"{config.data_root}/sky/download/swire"
    os.makedirs(dest_dir, exist_ok=True)
    fields = list(FIELDS.items())
    with progress_module.Stage("sky.download.swire") as st:
        for field_name, table in FIELDS_70.items():
            slug = field_name.lower().replace("-", "")
            dest_path = f"{dest_dir}/swire70_{slug}.csv"
            n_rows, n_bytes = _tap.query_csv(table, COLUMNS_70, "cntr", dest_path)
            print(f"swire build: {field_name} ({table}) -> {dest_path}: {n_rows} rows, {n_bytes} bytes")
        for i, (field_name, table) in enumerate(fields):
            slug = field_name.lower().replace("-", "")
            dest_path = f"{dest_dir}/swire_{slug}.csv"
            n_rows, n_bytes = _tap.query_csv(table, COLUMNS, "cntr", dest_path)
            print(f"swire build: {field_name} ({table}) -> {dest_path}: {n_rows} rows, {n_bytes} bytes")
            st.tick(i + 1, len(fields), "queries")
        st.done(dest_dir, fields=len(fields))


if __name__ == "__main__":
    build_module.run(build)
