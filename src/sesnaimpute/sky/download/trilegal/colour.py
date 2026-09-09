"""One TRILEGAL query in a photometric system that carries BOTH a Gaia G
and a 2MASS Ks (`build.py`'s own 2MASS+Spitzer query carries neither
Gaia band; TRILEGAL's form offers no system with Gaia, 2MASS and Spitzer
together -- the form's own 103 photometric-system tables were read: every
Gaia-bearing system drops Spitzer, W42).

A star's G - Ks at given (log T_eff, log g, [M/H]) is atmosphere physics,
independent of sky position, so one query anywhere gives the relation
`population.field_stars` needs to read a Gaia proxy colour for every
region's own stars, in place of the fitter's atmosphere-library templates
(SPEC_BMSTP_DRAFT.md 5.1, `N^{model->obs}`). The query runs at
`sky.download.trilegal.build.REGION_POINTINGS["Perseus"]`'s centre -- a
sparse, low-extinction, high-latitude field (b = -19.3 deg) whose stars
span the full range of temperature, gravity and metallicity a query needs
to populate the colour table's cells -- reusing `build.read_form`,
`build._query_fields` (same `mag_lim`, `icm_lim`, IMF and
internal-extinction settings as the survey's own query) and
`build.fetch_region_part`'s query/poll/fetch machinery, including its
busy-server retry (`build.submit_query`).

Photometric system `tab_mag_odfnew/tab_mag_gaiaDR2_tycho2_2mass.dat`
(Gaia DR2 + Tycho2 + 2MASS, Vega mags; Gaia passbands Evans et al. 2018,
A&A 616, A4) -- the only one of the form's systems carrying both a Gaia G
and a 2MASS Ks. The survey's own Gaia crossmatch
(`sky.download.gaia_crossmatch`) is DR3; the DR2-to-DR3 G passband change
is <= 0.03 mag (Riello et al. 2021, A&A 649, A3, section 5), well under
the mag of colour this table corrects, so DR2's G is read as DR3's G here.

Saved verbatim as `<data_root>/sky/download/trilegal/colour_gaiaDR2_2mass_perseus.dat`;
skipped if already present (the bytes are external and verbatim; delete
the file to draw a new realisation).
"""

import os
import time

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.sky.download.trilegal import build as trilegal_build

#: The only form system carrying both a Gaia G and a 2MASS Ks (module
#: docstring).
PHOTSYS_FILE = "tab_mag_odfnew/tab_mag_gaiaDR2_tycho2_2mass.dat"

#: Solid angle for this one query: `build`'s own 2MASS+Spitzer query at
#: Perseus (icm_lim/mag_lim reused here) returns ~29,200 raw stars per
#: deg2 there; this area targets ~10^5 stars from this system's own
#: depth, at or under the form's own 10 deg2 per-query cap and the
#: historical per-part size (Perseus's own build splits into 3.8 deg2
#: parts) already known to finish inside the form's 10-minute CPU limit.
AREA_DEG2 = 3.0

DEST_NAME = "colour_gaiaDR2_2mass_perseus.dat"


def build(config, regions=None):
    """Runs the one colour-relation query and saves it verbatim.
    `regions` is accepted for the common build entry-point convention
    (`sesnaimpute.build.run`) but ignored -- this is one fixed file, not a
    per-region product.
    """
    with progress_module.Stage("sky.download.trilegal.colour") as st:
        dest_dir = f"{config.data_root}/sky/download/trilegal"
        dest_path = f"{dest_dir}/{DEST_NAME}"
        if os.path.exists(dest_path):
            print(f"trilegal colour: {dest_path} present, skipped "
                  f"(delete it to draw a new realisation)")
            st.done(dest_path, skipped=1)
            return

        version, action_url, defaults = trilegal_build.read_form()
        print(f"trilegal colour: form version {version!r}, POST target {action_url}")

        info = trilegal_build.REGION_POINTINGS["Perseus"]
        fields = trilegal_build._query_fields(defaults, info["l_deg"], info["b_deg"], AREA_DEG2)
        fields["photsys_file"] = PHOTSYS_FILE  # the one override on top of build's own settings

        output_url, status_fields, submitted_at = trilegal_build.submit_query(action_url, fields)
        trilegal_build.wait_until_finished(action_url, status_fields)
        data = trilegal_build.fetch_output(output_url)
        wall_time_s = time.time() - submitted_at

        os.makedirs(dest_dir, exist_ok=True)
        with open(dest_path, "wb") as out:
            out.write(data)
        print(f"trilegal colour: l={info['l_deg']:.4f}, b={info['b_deg']:.4f}, "
              f"area={AREA_DEG2} deg2 -> {dest_path} ({len(data)} bytes, {wall_time_s:.0f}s)")
        st.done(dest_path, bytes=len(data), area_deg2=AREA_DEG2, wall_s=wall_time_s)


if __name__ == "__main__":
    run(build)
