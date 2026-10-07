"""The one table every catalog, sky, population, bmstp and atlas writer
reads from, and the one table `build.attrs`'s in-place pass walks
(CODING_RULES_BMSTP.md rule 5, briefs/ATTRS.md): `UNITS` and `READING`
for every dataset those writers create, keyed by the product's own file
stem (`<quantity>_<source>_<granule>`, `config.product_path`'s own name
for the file with any `__<Region>` suffix and `.hdf5` removed) and the
dataset's name. `sesnaimpute.fittp` carries its own readings (LOGFLUX)
and is not in this table.

A writer calls `build.write_dataset(group, name, data, *REGISTRY[(STEM,
name)])`, `STEM` being its own module-level constant. The migration pass
derives `STEM` from each file name on disk and raises, naming the file and
dataset, when a dataset it finds has no entry here.
"""

REGISTRY = {

    # catalog.curated -- sources_sesna_source: the per-region curated SESNA
    # catalogue, one row per source.
    ("sources_sesna_source", "NAME"): ("source name",
        "The source's name from the SESNA catalog, in this file's row order."),
    ("sources_sesna_source", "RA_DEG"): ("deg",
        "Right ascension, equinox J2000."),
    ("sources_sesna_source", "DEC_DEG"): ("deg",
        "Declination, equinox J2000."),
    ("sources_sesna_source", "GAL_L_DEG"): ("deg",
        "Galactic longitude."),
    ("sources_sesna_source", "GAL_B_DEG"): ("deg",
        "Galactic latitude."),
    ("sources_sesna_source", "CLASS"): ("SESNA class code",
        "SESNA's own classification for this source: 0 deeply embedded protostar, "
        "1 class I, 2 class II, 3 transition disk, 9 H2 shock blob, 19 PAH emitter "
        "(star-forming galaxy), 29 AGN, 39 PAH-contaminated source, 49 generic "
        "galaxy, 99 diskless star, -100 unclassified. No other dataset in this "
        "file depends on it."),
    ("sources_sesna_source", "AK_SESNA"): ("mag A_K",
        "The K-band extinction magnitude recorded for this source in the SESNA "
        "delivery."),
    ("sources_sesna_source", "FNU_MJY"): ("mJy",
        "The flux in each of the eight bands, in the band order this file's BANDS "
        "attribute gives. Where there is no detection the value is filled: the "
        "2MASS bands (J, H, Ks) take the survey's fixed detection bound, and the "
        "three IRAC bands and the one MIPS band take the source's own "
        "90%-completeness flux, or, where it has none, the median "
        "90%-completeness flux of its five nearest sky neighbours that do. "
        "ORIGIN_FNU in this file says which bands were filled and how."),
    ("sources_sesna_source", "SIGMA_FNU_MJY"): ("mJy",
        "The 1-sigma flux uncertainty for a measured detection in FNU_MJY, on the "
        "same mJy scale. Where FNU_MJY was filled instead of measured (see "
        "ORIGIN_FNU), this is the fixed value 0.99: a flag the SED fitter reads as "
        "an upper limit, not a real measurement uncertainty."),
    ("sources_sesna_source", "DCOMP90_MJY"): ("mJy",
        "The flux a source at this position and band would need to be detected at "
        "90% completeness: the survey's fixed bound for the 2MASS bands (J, H, "
        "Ks), this source's own 90%-completeness flux for the three IRAC bands "
        "and the one MIPS band, or, where it has none, the median "
        "90%-completeness flux of its five nearest sky neighbours that do."),
    ("sources_sesna_source", "ORIGIN_FNU"): ("code",
        "How FNU_MJY and DCOMP90_MJY in each band were obtained, one integer per "
        "band: 1 a measured detection, 2 the fixed 2MASS survey detection bound, "
        "90 the source's own 90%-completeness flux, 91 the median "
        "90%-completeness flux of its nearest sky neighbours."),

    # catalog.coverage -- coverage_sesna_hpx512: the catalogue's own observed
    # IRAC footprint on the nside-512 grid, per region.
    ("coverage_sesna_hpx512", "HPX_PIX"): ("nested HEALPix pixel, nside 512",
        "The region's admitted nside-512 pixel numbers (nested ordering), "
        "ascending. Row i here is row i of FRAC."),
    ("coverage_sesna_hpx512", "FRAC"): ("dimensionless",
        "For the matching pixel in HPX_PIX, the fraction of its 16 nside-2048 "
        "child pixels holding at least one SESNA source with a measured 3.6, "
        "4.5, 5.8 or 8.0 micron detection. This is the catalogue's own observed "
        "footprint, used as that pixel's survey coverage fraction."),

    # catalog.depth_grid, per-source product -- limits_sesna_source: each
    # source's own 50%-completeness flux limit, by band.
    ("limits_sesna_source", "F_LIM_50_MJY"): ("mJy",
        "For every source and each of the eight bands (J, H, Ks, 3.6, 4.5, 5.8, "
        "8.0 and 24 micron, this file's own band order), the flux at which a "
        "source at this source's own depth would be 50% likely to be detected. "
        "It is the region's fitted 50%-completeness flux for that band, shifted "
        "by how much brighter or fainter this source's own completeness-limit "
        "flux is than the region's typical source."),
    ("limits_sesna_source", "W_DEX"): ("dex",
        "One value per band (same eight bands and order as F_LIM_50_MJY), the "
        "roll-off width of the region's fitted detection curve: how many factors "
        "of 10 in flux the detection fraction takes to fall from high to low "
        "around F_50_REGION_MJY. The same eight values apply to every source in "
        "this file."),
    ("limits_sesna_source", "F_50_REGION_MJY"): ("mJy",
        "One value per band (same eight bands and order as F_LIM_50_MJY), the "
        "region's own fitted 50%-completeness flux: the flux at which half of "
        "the region's sources at that depth are detected. The same eight values "
        "apply to every source in this file. A value of +inf means the region "
        "did not sample enough of that band's roll-off to fit one."),
    ("limits_sesna_source", "ALPHA_REGION"): ("dimensionless",
        "One value per band (same eight bands and order as F_LIM_50_MJY), the "
        "power-law slope of the region's own source counts at fluxes well above "
        "F_50_REGION_MJY, from the same fit. The same eight values apply to "
        "every source in this file."),
    ("limits_sesna_source", "DCOMP90_REF_LOG10"): ("log10 mJy",
        "One value per band (same eight bands and order as F_LIM_50_MJY), the "
        "log10 90%-completeness flux that F_LIM_50_MJY's per-source shift is "
        "measured from: the median, over the region's own low-extinction, "
        "well-covered sources, of log10 DCOMP90_MJY (catalog/curated's own "
        "column). The same eight values apply to every source in this file."),
    ("limits_sesna_source", "LIMIT_KIND"): ("code",
        "One value per band (same eight bands and order as F_LIM_50_MJY), how "
        "F_50_REGION_MJY for that band was obtained: \"fit\" means the region's "
        "own source counts show a genuine roll-off; \"bound\" means this band's "
        "faint end is set by a different, more restrictive band's requirement, "
        "so F_50_REGION_MJY is a bound inherited from that other band rather "
        "than this band's own turnover; \"unsurveyed\" means the region has too "
        "few detections in that band to fit a roll-off at all, and "
        "F_50_REGION_MJY is +inf. The same eight values apply to every source "
        "in this file."),

    # catalog.depth_grid, per-pixel product -- depth-grid_sesna_hpx512: the
    # same per-source limits, summarised on the nside-512 grid.
    ("depth-grid_sesna_hpx512", "HPX_PIX_512"): ("nested HEALPix pixel, nside 512",
        "The region's admitted nside-512 pixel numbers (nested ordering), "
        "ascending. Row i here matches row i of every other dataset in this "
        "file."),
    ("depth-grid_sesna_hpx512", "N_SOURCES"): ("sources",
        "How many SESNA sources fall in the matching pixel in HPX_PIX_512. A "
        "value of 0 means the pixel holds no source of its own, and its other "
        "values in this file are copied from its nearest pixel that does."),
    ("depth-grid_sesna_hpx512", "F_LIM_50_MED_MJY"): ("mJy",
        "For the matching pixel in HPX_PIX_512 and each of the eight bands (J, "
        "H, Ks, 3.6, 4.5, 5.8, 8.0 and 24 micron, this file's own band order), "
        "the median, over the pixel's own sources, of each source's own "
        "50%-completeness flux limit (catalog/sesna's per-source limits "
        "product). A pixel with no sources of its own (N_SOURCES = 0) takes its "
        "nearest occupied pixel's value."),
    ("depth-grid_sesna_hpx512", "F_LIM_50_PIX_MJY"): ("mJy",
        "For the matching pixel in HPX_PIX_512 and each of the eight bands (J, "
        "H, Ks, 3.6, 4.5, 5.8, 8.0 and 24 micron, this file's own band order), "
        "the median, over the pixel's own sources, of each source's own "
        "50%-completeness flux limit (catalog/sesna's per-source limits "
        "product). A pixel with no sources of its own (N_SOURCES = 0) takes its "
        "nearest occupied pixel's value."),
    ("depth-grid_sesna_hpx512", "W_DEX_PIX"): ("dex",
        "For the matching pixel in HPX_PIX_512 and each of the eight bands (J, "
        "H, Ks, 3.6, 4.5, 5.8, 8.0 and 24 micron, this file's own band order), "
        "the roll-off width of the region's fitted detection curve: how many "
        "factors of 10 in flux the detection fraction takes to fall from high "
        "to low around that band's 50%-completeness flux. The same eight values "
        "apply to every pixel in this file."),

}
