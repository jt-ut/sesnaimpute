"""The Gaia DR3 published position-dependent selection function's one
input map: Cantat-Gaudin et al. 2023 (A&A 669, A55; arXiv 2208.09335),
`M10`, the median G magnitude of catalogued sources with
`astrometric_matched_transits <= 10` in a HEALPix order-7 (nside 128)
patch -- the map's own proxy for how deep and how crowded a patch is,
read against eq. 2-4's own fitted relation between (x, y, z) and M10 to
give the completeness curve's own shape at that patch (C7 brief Part A).
Fetches `allsky_M10_hpx7.hdf5` verbatim from the paper's own Zenodo
record (DOI 10.5281/zenodo.8063930).

Feeds `population.anchor_tiles.gaia_selection`/`gaia_detection_weight`:
an unmatched Gaia source is now scored as a non-detection against this
map's own completeness at its sky pixel, replacing the single survey-
wide sigmoid (`GAIA_G_LIM_MAG`/`GAIA_G_ROLLOFF_MAG`) that gave every
unmatched source Gamma = 1 (no information) regardless of position.
"""

from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

URL = "https://zenodo.org/api/records/8063930/files/allsky_M10_hpx7.hdf5/content"
FILENAME = "allsky_M10_hpx7.hdf5"

#: HEALPix order 7, nside 128: `12 * nside**2` pixels, one row of the
#: file's own `data` dataset per pixel (module docstring).
_NSIDE = 128
_N_PIX_EXPECT = 12 * _NSIDE ** 2


def build(config, regions=None):
    """Fetches `allsky_M10_hpx7.hdf5` verbatim into
    `sky/download/gaia_selection/`. `regions` is accepted for interface
    uniformity and ignored: this is a survey-wide, all-sky product. A
    file already present is skipped (`_fetch.fetch`). Checks the file's
    own pixel count against the order-7 map's expectation -- the one
    structural fact this download can verify without opening a
    nested/ring question (`population.anchor_tiles.gaia_selection`
    settles that one, against a known dense field, not here)."""
    dest_path = f"{config.data_root}/sky/download/gaia_selection/{FILENAME}"
    with progress_module.Stage("sky.download.gaia_selection") as st:
        fetch(URL, dest_path)
        import h5py
        with h5py.File(dest_path, "r") as f:
            n_pix = f["data"].shape[0]
        if n_pix != _N_PIX_EXPECT:
            raise ValueError(
                "sky.download.gaia_selection: %s holds %d rows, not the order-7 "
                "map's 12*128^2 = %d -- wrong file or a different HEALPix order"
                % (dest_path, n_pix, _N_PIX_EXPECT))
        st.done(dest_path, files=1, n_pix=n_pix)


if __name__ == "__main__":
    run(build)
