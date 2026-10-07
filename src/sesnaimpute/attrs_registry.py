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

    # sky.derived.knots -- five whole-survey products (GRANULE "survey", no
    # region axis) feeding the H2-shock-knot rate (SPEC_PRIORS.md section 7).
    ("giannini2013_knots_survey", "KNOT_ID"): ("knot identifier",
        "This knot's identifier in Giannini et al. (2013)."),
    ("giannini2013_knots_survey", "RA_DEG"): ("deg", "Right ascension, equinox J2000."),
    ("giannini2013_knots_survey", "DEC_DEG"): ("deg", "Declination, equinox J2000."),
    ("giannini2013_knots_survey", "SIZE_ARCSEC2"): ("arcsec^2",
        "The knot's projected area on the sky, from Giannini et al. (2013)."),
    ("giannini2013_knots_survey", "FLUX_2P12_MJY"): ("mJy",
        "The knot's 2.12 micron (H2 1-0 S(1)) line flux density, from Giannini "
        "et al. (2013)."),
    ("giannini2013_knots_survey", "FLUX_ERR_2P12_MJY"): ("mJy",
        "The 1-sigma uncertainty on FLUX_2P12_MJY."),
    ("giannini2013_knots_survey", "LUMINOSITY_IRAC_1E-2LSUN"): ("1e-2 Lsun",
        "The knot's summed-IRAC luminosity: multiply by 0.01 to get solar "
        "luminosities."),
    ("giannini2013_knots_survey", "LUMINOSITY_2P12_1E-2LSUN"): ("1e-2 Lsun",
        "The knot's 2.12 micron line luminosity: multiply by 0.01 to get solar "
        "luminosities."),
    ("giannini2013_knots_survey", "FLUX_I1_MJY"): ("mJy",
        "The knot's flux density in the Spitzer IRAC 3.6 micron band, from "
        "Giannini et al. (2013)."),
    ("giannini2013_knots_survey", "FLUX_ERR_I1_MJY"): ("mJy",
        "The 1-sigma uncertainty on FLUX_I1_MJY."),
    ("giannini2013_knots_survey", "FLUX_I2_MJY"): ("mJy",
        "The knot's flux density in the Spitzer IRAC 4.5 micron band, from "
        "Giannini et al. (2013)."),
    ("giannini2013_knots_survey", "FLUX_ERR_I2_MJY"): ("mJy",
        "The 1-sigma uncertainty on FLUX_I2_MJY."),
    ("giannini2013_knots_survey", "FLUX_I3_MJY"): ("mJy",
        "The knot's flux density in the Spitzer IRAC 5.8 micron band, from "
        "Giannini et al. (2013)."),
    ("giannini2013_knots_survey", "FLUX_ERR_I3_MJY"): ("mJy",
        "The 1-sigma uncertainty on FLUX_I3_MJY."),
    ("giannini2013_knots_survey", "FLUX_I4_MJY"): ("mJy",
        "The knot's flux density in the Spitzer IRAC 8.0 micron band, from "
        "Giannini et al. (2013)."),
    ("giannini2013_knots_survey", "FLUX_ERR_I4_MJY"): ("mJy",
        "The 1-sigma uncertainty on FLUX_I4_MJY."),
    ("giannini2013_knots_survey", "UPPER_LIMIT_I1"): ("boolean",
        "True where the 3.6 micron flux above is an upper limit rather than a "
        "measured detection, Giannini et al. (2013)'s own flag."),
    ("giannini2013_knots_survey", "UPPER_LIMIT_I2"): ("boolean",
        "True where the 4.5 micron flux above is an upper limit rather than a "
        "measured detection. Giannini et al. (2013) reports no upper-limit flag "
        "for this band, so this is always False."),
    ("giannini2013_knots_survey", "UPPER_LIMIT_I3"): ("boolean",
        "True where the 5.8 micron flux above is an upper limit rather than a "
        "measured detection, Giannini et al. (2013)'s own flag."),
    ("giannini2013_knots_survey", "UPPER_LIMIT_I4"): ("boolean",
        "True where the 8.0 micron flux above is an upper limit rather than a "
        "measured detection, Giannini et al. (2013)'s own flag."),

    ("davis2009_knots_survey", "KNOT_ID"): ("knot identifier",
        "This knot's identifier in Davis et al. (2009)."),
    ("davis2009_knots_survey", "RA_DEG"): ("deg", "Right ascension, equinox J2000."),
    ("davis2009_knots_survey", "DEC_DEG"): ("deg", "Declination, equinox J2000."),
    ("davis2009_knots_survey", "TANGENTIAL_VELOCITY_KM_S"): ("km/s",
        "The knot's proper-motion tangential velocity, from Davis et al. (2009)."),
    ("davis2009_knots_survey", "POSITION_ANGLE_DEG"): ("deg",
        "The position angle of the knot's proper-motion vector, east of north, "
        "from Davis et al. (2009)."),
    ("davis2009_knots_survey", "QUALITY_FLAG"): ("code",
        "Davis et al. (2009)'s own quality flag for this proper-motion "
        "measurement: 0 good, 2 uncertain."),

    ("walawender2005_knots_survey", "FIELD"): ("field name",
        "Which field this object was catalogued in: Perseus or Barnard 1."),
    ("walawender2005_knots_survey", "TABLE_ORIGIN"): ("table label",
        "Which table of Walawender et al. (2005) this row comes from: "
        "hh_known (previously known Herbig-Haro objects, Perseus), hh_new "
        "(newly discovered Herbig-Haro objects, Perseus), or h2_shocks (H2 "
        "narrowband-selected shocks, Barnard 1)."),
    ("walawender2005_knots_survey", "H2_SELECTED"): ("boolean",
        "True where the object was selected by narrowband H2 imaging "
        "(Barnard 1's h2_shocks table only); False for Perseus's optical "
        "Halpha/[S II] Herbig-Haro objects."),
    ("walawender2005_knots_survey", "DESIGNATION"): ("object name",
        "The object's designation in its own source table."),
    ("walawender2005_knots_survey", "RA_DEG"): ("deg", "Right ascension, equinox J2000."),
    ("walawender2005_knots_survey", "DEC_DEG"): ("deg", "Declination, equinox J2000."),

    ("uwish2_knots_survey", "UWISH2_ID"): ("UWISH2 catalogue identifier",
        "This feature's identifier in the UWISH2 catalogue (Froebrich et al. "
        "2015)."),
    ("uwish2_knots_survey", "RA_DEG"): ("deg", "Right ascension, equinox J2000."),
    ("uwish2_knots_survey", "DEC_DEG"): ("deg", "Declination, equinox J2000."),
    ("uwish2_knots_survey", "AREA_ARCSEC2"): ("arcsec^2",
        "The feature's projected area on the sky."),
    ("uwish2_knots_survey", "RADIUS_ARCSEC"): ("arcsec",
        "The feature's equivalent circular radius."),
    ("uwish2_knots_survey", "SURFACE_BRIGHTNESS_MEDIAN_1E-19_W_M2_ARCSEC2"): ("1e-19 W m^-2 arcsec^-2",
        "The feature's aperture-median H2 1-0 S(1) surface brightness: "
        "multiply by 1e-19 to get W per square meter per square arcsecond."),
    ("uwish2_knots_survey", "SURFACE_BRIGHTNESS_MAX_1E-19_W_M2_ARCSEC2"): ("1e-19 W m^-2 arcsec^-2",
        "The feature's peak H2 1-0 S(1) surface brightness: multiply by 1e-19 "
        "to get W per square meter per square arcsecond."),
    ("uwish2_knots_survey", "TOTAL_FLUX_1E-19_W_M2"): ("1e-19 W m^-2",
        "The feature's total H2 1-0 S(1) line flux, summed over its aperture: "
        "multiply by 1e-19 to get W per square meter."),
    ("uwish2_knots_survey", "CLASS"): ("code",
        "Froebrich et al. (2015)'s own one-letter classification for this "
        "feature: j jet, p planetary nebula, s supernova remnant, u unknown."),
    ("uwish2_knots_survey", "JET_CLASS"): ("boolean",
        "True where CLASS is \"j\" (jet)."),
    ("uwish2_knots_survey", "GROUP"): ("group identifier",
        "Froebrich et al. (2015)'s own grouping identifier linking features "
        "that belong to the same physical outflow."),

    ("uwish2_images_knots_survey", "TILE"): ("tile name",
        "Which UWISH2 survey tile this image belongs to."),
    ("uwish2_images_knots_survey", "IMAGE"): ("image name",
        "This WFCAM detector-array image's own identifier."),
    ("uwish2_images_knots_survey", "RA_DEG"): ("deg",
        "Right ascension of the image centre, equinox J2000."),
    ("uwish2_images_knots_survey", "DEC_DEG"): ("deg",
        "Declination of the image centre, equinox J2000."),
    ("uwish2_images_knots_survey", "GLON_DEG"): ("deg",
        "Galactic longitude of the image centre."),
    ("uwish2_images_knots_survey", "GLAT_DEG"): ("deg",
        "Galactic latitude of the image centre."),
    ("uwish2_images_knots_survey", "NOISE"): ("counts",
        "The image's own one-pixel background noise level, in the map's "
        "native calibrated counts (Froebrich et al. 2015, their Table C1)."),

    ("colours_knots_survey", "LOG10_RATIO"): ("dex",
        "log10(F_IRAC_band / F_2.12um) for every Giannini et al. (2013) knot "
        "with a measured, non-upper-limit flux in both the 2.12 micron line "
        "and this group's own IRAC band, one value per knot. This dataset's "
        "enclosing group is named by the Spitzer IRAC band: I1 3.6 micron, I2 "
        "4.5 micron, I3 5.8 micron, I4 8.0 micron."),

}
