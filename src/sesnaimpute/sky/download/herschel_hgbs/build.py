"""The Herschel Gould Belt Survey column-density maps.

Andre P. et al. 2010, A&A 518, L102 (survey description); per-cloud data
releases at herschel.fr (Gould Belt archive) and the CEA Gould Belt
archive. Fetches the 23 per-cloud/per-field column-density FITS maps
(H2 column density from modified-blackbody fits to the Herschel
70-500 um bands) verbatim, upstream file names kept.

Feeds SPEC_PRIORS.md section 1.1 (columns; the Herschel-covered branch of
the adopted dust column, C6 "Herschel where covered").
"""

from sesnaimpute.build import run
from sesnaimpute.sky.download._fetch import fetch

# filename -> upstream URL, lifted from the old
# fetch_external.herschel_hgbs.build.maps acquisition table. The three
# `cham*` maps are smallest by upstream byte size and lead the dict so
# the rehearsal knob below pulls a small file first.
_FILES = {
    "chamIII-coldens.fits.gz": "http://www.herschel.fr/cea/gouldbelt/en/archives/chamaeleonIII/chamIII-coldens.fits.gz",
    "chamII-coldens.fits.gz": "http://www.herschel.fr/cea/gouldbelt/en/archives/chamaeleonII/chamII-coldens.fits.gz",
    "chamI-coldens.fits.gz": "http://www.herschel.fr/cea/gouldbelt/en/archives/chamaeleonI/chamI-coldens.fits.gz",
    "HGBS_aquilaM2_column_density_map.fits.gz": "http://www.herschel.fr/Images/astImg/66/HGBS_aquilaM2_column_density_map.fits.gz",
    "HGBS_cep1157_column_density_map.fits": "http://www.herschel.fr/Images/astImg/66/HGBS_cep1157_column_density_map.fits",
    "HGBS_cep1172_column_density_map.fits": "http://www.herschel.fr/Images/astImg/66/HGBS_cep1172_column_density_map.fits",
    "HGBS_cep1228_column_density_map.fits": "http://www.herschel.fr/Images/astImg/66/HGBS_cep1228_column_density_map.fits",
    "HGBS_cep1241_column_density_map.fits": "http://www.herschel.fr/Images/astImg/66/HGBS_cep1241_column_density_map.fits",
    "HGBS_cep1251_column_density_map.fits": "http://www.herschel.fr/Images/astImg/66/HGBS_cep1251_column_density_map.fits",
    "HGBS_craNS_column_density_map.fits.gz": "http://www.herschel.fr/Images/astImg/66/HGBS_craNS_column_density_map.fits.gz",
    "HGBS_ic5146_column_density_map.fits.gz": "http://www.herschel.fr/Images/astImg/66/HGBS_ic5146_column_density_map.fits.gz",
    "HGBS_lupIII_column_density_map.fits.gz": "http://www.herschel.fr/Images/astImg/66/HGBS_lupIII_column_density_map.fits.gz",
    "HGBS_lupIV_column_density_map.fits.gz": "http://www.herschel.fr/Images/astImg/66/HGBS_lupIV_column_density_map.fits.gz",
    "HGBS_lupI_column_density_map.fits.gz": "http://www.herschel.fr/Images/astImg/66/HGBS_lupI_column_density_map.fits.gz",
    "HGBS_musca_column_density_map.fits.gz": "http://www.herschel.fr/Images/astImg/66/HGBS_musca_column_density_map.fits.gz",
    "HGBS_oph_l1688_column_density_map.fits.gz": "http://www.herschel.fr/Images/astImg/66/HGBS_oph_l1688_column_density_map.fits.gz",
    "HGBS_orionA_column_density_map.fits.gz": "http://www.herschel.fr/Images/astImg/66/HGBS_orionA_column_density_map.fits.gz",
    "HGBS_orionB_column_density_map.fits.gz": "http://www.herschel.fr/Images/astImg/66/HGBS_orionB_column_density_map.fits.gz",
    "HGBS_perseus_column_density_map.fits.gz": "http://www.herschel.fr/Images/astImg/66/HGBS_perseus_column_density_map.fits.gz",
    "HGBS_pipe_column_density_map.fits.gz": "http://www.herschel.fr/Images/astImg/66/HGBS_pipe_column_density_map.fits.gz",
    "HGBS_serpens_column_density_map.fits.gz": "http://www.herschel.fr/Images/astImg/66/HGBS_serpens_column_density_map.fits.gz",
    "HGBS_taurusTMC1_column_density_map.fits.gz": "http://www.herschel.fr/Images/astImg/66/HGBS_taurusTMC1_column_density_map.fits.gz",
    "HGBS_taurus_L1495_column_density_map.fits.gz": "http://www.herschel.fr/Images/astImg/66/HGBS_taurus_L1495_column_density_map.fits.gz",
}


def build(config, regions=None, _limit=None):
    """Fetches each of the 23 HGBS column-density maps, verbatim.
    `regions` is accepted for interface uniformity and ignored: the map
    set is fixed by the survey's own field list, not by SESNA's regions.
    `_limit` is a rehearsal knob: when set, only the first `_limit`
    files are fetched, smallest file first by upstream byte size.
    """
    dest_dir = f"{config.data_root}/sky/download/herschel_hgbs"
    names = list(_FILES)
    if _limit is not None:
        names = names[:_limit]
    for name in names:
        fetch(_FILES[name], f"{dest_dir}/{name}")


if __name__ == "__main__":
    run(build)
