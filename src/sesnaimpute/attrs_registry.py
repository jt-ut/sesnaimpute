"""The one table every catalog, sky, population, bmstp, fittp and atlas
writer reads from, and the one table `build.attrs`'s in-place pass walks
(CODING_RULES_BMSTP.md rule 5, briefs/ATTRS.md): `UNITS` and `READING`
for every dataset those writers create, keyed by the product's own file
stem (`<quantity>_<source>_<granule>`, `config.product_path`'s own name
for the file with any `__<Region>` suffix and `.hdf5` removed) and the
dataset's name. `fittp.atlas`'s `MEAN_P_<CLASS>` readings are generated
from a sentence template at that module's own call site (READINGS brief
section 3) and stay there, not here -- `fittp.atlas`'s other three
datasets (`HPX_PIX_512`, `N_SOURCES`, `N_YSO_ABOVE_HALF`) ride along with
them, so `posterior_atlas_hpx512` carries no entry in this table either.

A writer calls `build.write_dataset(group, name, data, *REGISTRY[(STEM,
name)])`, `STEM` being its own module-level constant (or, where one
reading applies to several files of the same shape -- `fittp.sweep`'s six
per-class fit files, `fittp.emission`'s six per-library tables -- computed
the same way `config.product_path` would for that file, inside a loop
over the fixed class/library list at the bottom of this module). The
migration pass derives `STEM` from each file name on disk and raises,
naming the file and dataset, when a dataset it finds has no entry here.
"""

REGISTRY = {

    # catalog.curated -- sources_sesna_source: the per-region curated SESNA
    # catalog, one row per source.
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

    # population.young_stars -- young-stars_anchors_hpx512: the model's own
    # expected young-star count in each STAR anchor pixel, by magnitude,
    # subtracted from the observed Gaia/2MASS histograms before fitting
    # the field-star weight.
    ("young-stars_anchors_hpx512", "HPX_PIX_512"): ("nested HEALPix pixel, nside 512",
        "This region's own occupied nside-512 pixel numbers (nested "
        "ordering). Row i here is row i of N_YOUNG_TOTAL."),
    ("young-stars_anchors_hpx512", "N_G_YOUNG"): ("stars",
        "For the matching pixel in HPX_PIX_512 and each Gaia G magnitude "
        "bin of the STAR anchor's own half-magnitude binning grid, the "
        "model's expected number of 1-Myr young stars at that apparent "
        "magnitude, weighted by the Gaia detection probability at that "
        "magnitude and summed to N_YOUNG_TOTAL for the pixel."),
    ("young-stars_anchors_hpx512", "N_KS_YOUNG"): ("stars",
        "For the matching pixel in HPX_PIX_512 and each 2MASS Ks "
        "magnitude bin of the STAR anchor's own half-magnitude binning "
        "grid, the model's expected number of 1-Myr young stars at that "
        "apparent magnitude, summed to N_YOUNG_TOTAL for the pixel "
        "together with the stars fainter than that grid's own faint "
        "edge and the stars above the model's own 1.4 solar mass top."),
    ("young-stars_anchors_hpx512", "N_YOUNG_TOTAL"): ("stars",
        "For the matching pixel in HPX_PIX_512, the model's total expected "
        "number of young stars belonging to this region's cloud: the "
        "young-star law integrated over the pixel's own column map, times "
        "the pixel's solid angle, times the square of the cloud's own "
        "share of the line-of-sight column."),

    # population.yso -- prior_yso_sightline: the young-star embedding
    # density along each occupied sightline of a region.
    ("prior_yso_sightline", "HPX_PIX_256"): ("nested HEALPix pixel, nside 256",
        "This region's own occupied nside-256 pixel numbers (nested "
        "ordering). Row i here is row i of XI_EDGES and P_XI."),
    ("prior_yso_sightline", "XI_EDGES"): ("dimensionless",
        "For the matching pixel in HPX_PIX_256, the edges, in u = "
        "(cumulative extinction to a depth) / (the sightline's own total "
        "extinction), of the 32 equal-mass cells P_XI is tabulated on: "
        "cell j runs from XI_EDGES[:, j] to XI_EDGES[:, j+1], with u = 0 "
        "at the observer and u = 1 at the sightline's own total column."),
    ("prior_yso_sightline", "P_XI"): ("per unit u",
        "For the matching pixel in HPX_PIX_256 and each cell of XI_EDGES, "
        "the young-star embedding density per unit u: the probability "
        "that a young star on this sightline sits in that cell, divided "
        "by the cell's own width in u. Integrating P_XI times the cell "
        "width over all cells gives 1."),

    # bmstp.atlas -- prior_atlas_hpx512: the prior atlas, per admitted
    # nside-512 pixel, every class's cataloged and intrinsic density and
    # their grid breakdowns.
    ("prior_atlas_hpx512", "HPX_PIX_512"): ("nested HEALPix pixel, nside 512",
        "This region's own admitted nside-512 pixel numbers (nested "
        "ordering). Row i here is row i of every other per-pixel dataset "
        "in this file."),
    ("prior_atlas_hpx512", "A_COL_K"): ("mag A_K",
        "For the matching pixel in HPX_PIX_512, the adopted extinction "
        "column every class's density is dimmed by."),
    ("prior_atlas_hpx512", "COVERAGE"): ("dimensionless",
        "For the matching pixel in HPX_PIX_512, the catalog's own "
        "observed footprint fraction (the catalog coverage product): "
        "what share of this pixel SESNA was actually extracted on."),
    ("prior_atlas_hpx512", "F_LIM_50_PIX_MJY"): ("mJy",
        "For the matching pixel in HPX_PIX_512 and each of the eight "
        "bands (J, H, Ks, 3.6, 4.5, 5.8, 8.0 and 24 micron, this "
        "project's own band order), the pixel's own marginalized "
        "50%-completeness flux limit."),
    ("prior_atlas_hpx512", "N_CAT_STAR"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "the STAR population: its dimmed intrinsic density "
        "(INTENSITY_STAR) weighted by the probability of clearing "
        "SESNA's two-of-eight-band detection rule at this pixel's own "
        "depth."),
    ("prior_atlas_hpx512", "N_CAT_AGB"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "the AGB population: its dimmed intrinsic density weighted by "
        "the probability of clearing SESNA's two-of-eight-band detection "
        "rule at this pixel's own depth."),
    ("prior_atlas_hpx512", "N_CAT_PAHC"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "the PAHC population: its dimmed intrinsic density weighted by "
        "the probability of clearing SESNA's two-of-eight-band detection "
        "rule at this pixel's own depth."),
    ("prior_atlas_hpx512", "N_CAT_GAL"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "the background-galaxy population: its dimmed intrinsic density "
        "weighted by the probability of clearing SESNA's two-of-eight-"
        "band detection rule at this pixel's own depth."),
    ("prior_atlas_hpx512", "N_CAT_YSO"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "the YSO population: its dimmed intrinsic density weighted by "
        "the probability of clearing SESNA's two-of-eight-band detection "
        "rule at this pixel's own depth."),
    ("prior_atlas_hpx512", "N_CAT_H2S"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "the H2-shock population: its dimmed intrinsic density weighted "
        "by the probability of clearing SESNA's two-of-eight-band "
        "detection rule at this pixel's own depth."),
    ("prior_atlas_hpx512", "INTENSITY_STAR"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the STAR population's "
        "own dimmed intrinsic density at this pixel's own column, before "
        "the cataloged-detection probability N_CAT_STAR applies."),
    ("prior_atlas_hpx512", "INTENSITY_AGB"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the AGB population's own "
        "dimmed intrinsic density at this pixel's own column, before the "
        "cataloged-detection probability N_CAT_AGB applies."),
    ("prior_atlas_hpx512", "INTENSITY_PAHC"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the PAHC population's "
        "own dimmed intrinsic density at this pixel's own column, before "
        "the cataloged-detection probability N_CAT_PAHC applies."),
    ("prior_atlas_hpx512", "INTENSITY_GAL"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the background-galaxy "
        "population's own dimmed intrinsic density at this pixel's own "
        "column, before the cataloged-detection probability N_CAT_GAL "
        "applies."),
    ("prior_atlas_hpx512", "INTENSITY_YSO"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the YSO population's own "
        "dimmed intrinsic density at this pixel's own column, before the "
        "cataloged-detection probability N_CAT_YSO applies."),
    ("prior_atlas_hpx512", "INTENSITY_H2S"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the H2-shock "
        "population's own dimmed intrinsic density at this pixel's own "
        "column, after the region's own knot-rate convolution and before "
        "the cataloged-detection probability N_CAT_H2S applies."),
    ("prior_atlas_hpx512", "N_ABOVE_STAR"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, how much of the STAR "
        "population's own intrinsic density (INTENSITY_STAR) lies above "
        "this pixel's own I2 (4.5 micron) 50%-completeness flux, read "
        "directly from the class's own stored shape rather than from "
        "the cataloged-detection fraction."),
    ("prior_atlas_hpx512", "N_ABOVE_AGB"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, how much of the AGB "
        "population's own intrinsic density lies above this pixel's own "
        "I2 (4.5 micron) 50%-completeness flux, read directly from the "
        "class's own stored shape rather than from the cataloged-"
        "detection fraction."),
    ("prior_atlas_hpx512", "N_ABOVE_PAHC"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, how much of the PAHC "
        "population's own intrinsic density lies above this pixel's own "
        "8 micron depth cut, read directly from the class's own stored "
        "shape rather than from the cataloged-detection fraction."),
    ("prior_atlas_hpx512", "N_ABOVE_GAL"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, how much of the "
        "background-galaxy population's own intrinsic density lies "
        "above this pixel's own I2 (4.5 micron) 50%-completeness flux, "
        "read directly from the class's own stored shape rather than "
        "from the cataloged-detection fraction."),
    ("prior_atlas_hpx512", "N_ABOVE_YSO"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, how much of the YSO "
        "population's own intrinsic density lies above this pixel's own "
        "I2 (4.5 micron) 50%-completeness flux, read directly from the "
        "class's own stored shape rather than from the cataloged-"
        "detection fraction."),
    ("prior_atlas_hpx512", "N_ABOVE_H2S"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, how much of the "
        "H2-shock population's own intrinsic density lies above this "
        "pixel's own I2 (4.5 micron) 50%-completeness flux, read "
        "directly from the class's own stored shape rather than from "
        "the cataloged-detection fraction."),
    ("prior_atlas_hpx512", "N_CAT_BRIGHT3_STAR"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "STAR members predicted brighter than three times this pixel's "
        "own I2 50%-completeness limit, where completeness is "
        "effectively 1 on both the cataloged and the predicted side."),
    ("prior_atlas_hpx512", "N_CAT_BRIGHT3_AGB"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "AGB members predicted brighter than three times this pixel's "
        "own I2 50%-completeness limit, where completeness is "
        "effectively 1 on both the cataloged and the predicted side."),
    ("prior_atlas_hpx512", "N_CAT_BRIGHT3_PAHC"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "PAHC members predicted brighter than three times this pixel's "
        "own I2 50%-completeness limit, where completeness is "
        "effectively 1 on both the cataloged and the predicted side."),
    ("prior_atlas_hpx512", "N_CAT_BRIGHT3_GAL"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "background-galaxy members predicted brighter than three times "
        "this pixel's own I2 50%-completeness limit, where completeness "
        "is effectively 1 on both the cataloged and the predicted side."),
    ("prior_atlas_hpx512", "N_CAT_BRIGHT3_YSO"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "YSO members predicted brighter than three times this pixel's "
        "own I2 50%-completeness limit, where completeness is "
        "effectively 1 on both the cataloged and the predicted side."),
    ("prior_atlas_hpx512", "N_CAT_BRIGHT3_H2S"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "H2-shock members predicted brighter than three times this "
        "pixel's own I2 50%-completeness limit, where completeness is "
        "effectively 1 on both the cataloged and the predicted side."),
    ("prior_atlas_hpx512", "N_CAT_BRIGHT10_STAR"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "STAR members predicted brighter than ten times this pixel's "
        "own I2 50%-completeness limit, where completeness is "
        "effectively 1 on both the cataloged and the predicted side."),
    ("prior_atlas_hpx512", "N_CAT_BRIGHT10_AGB"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "AGB members predicted brighter than ten times this pixel's own "
        "I2 50%-completeness limit, where completeness is effectively 1 "
        "on both the cataloged and the predicted side."),
    ("prior_atlas_hpx512", "N_CAT_BRIGHT10_PAHC"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "PAHC members predicted brighter than ten times this pixel's own "
        "I2 50%-completeness limit, where completeness is effectively 1 "
        "on both the cataloged and the predicted side."),
    ("prior_atlas_hpx512", "N_CAT_BRIGHT10_GAL"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "background-galaxy members predicted brighter than ten times "
        "this pixel's own I2 50%-completeness limit, where completeness "
        "is effectively 1 on both the cataloged and the predicted side."),
    ("prior_atlas_hpx512", "N_CAT_BRIGHT10_YSO"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "YSO members predicted brighter than ten times this pixel's own "
        "I2 50%-completeness limit, where completeness is effectively 1 "
        "on both the cataloged and the predicted side."),
    ("prior_atlas_hpx512", "N_CAT_BRIGHT10_H2S"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "H2-shock members predicted brighter than ten times this "
        "pixel's own I2 50%-completeness limit, where completeness is "
        "effectively 1 on both the cataloged and the predicted side."),
    ("prior_atlas_hpx512", "N_CAT_CELL_GAL"): ("objects",
        "For the matching pixel in HPX_PIX_512, the background-galaxy "
        "population's own expected cataloged count on the common "
        "(log10 xi, log10 F_4.5) grid: summing every cell gives this "
        "pixel's own RATIO_GAL times the region's source count."),
    ("prior_atlas_hpx512", "N_CAT_CELL_STAR"): ("objects",
        "For the matching pixel in HPX_PIX_512, the STAR population's "
        "own expected cataloged count on the common (log10 xi, "
        "log10 F_4.5) grid: summing every cell gives this pixel's own "
        "RATIO_STAR times the region's source count."),
    ("prior_atlas_hpx512", "N_CAT_CELL_PAHC"): ("objects",
        "For the matching pixel in HPX_PIX_512, the PAHC population's "
        "own expected cataloged count on the common (log10 xi, "
        "log10 F_4.5) grid: summing every cell gives this pixel's own "
        "RATIO_PAHC times the region's source count."),
    ("prior_atlas_hpx512", "N_CAT_CELL_AGB"): ("objects",
        "For the matching pixel in HPX_PIX_512, the AGB population's own "
        "expected cataloged count on the common (log10 xi, log10 F_4.5) "
        "grid: summing every cell gives this pixel's own RATIO_AGB times "
        "the region's source count."),
    ("prior_atlas_hpx512", "N_CAT_CELL_YSO"): ("objects",
        "For the matching pixel in HPX_PIX_512, the YSO population's own "
        "expected cataloged count on the common (log10 xi, log10 F_4.5) "
        "grid: summing every cell gives this pixel's own RATIO_YSO times "
        "the region's source count."),
    ("prior_atlas_hpx512", "N_CAT_CELL_H2S"): ("objects",
        "For the matching pixel in HPX_PIX_512, the H2-shock "
        "population's own expected cataloged count on the common "
        "(log10 xi, log10 F_4.5) grid: summing every cell gives this "
        "pixel's own RATIO_H2S times the region's source count."),
    ("prior_atlas_hpx512", "N_CELL_STAR"): ("objects",
        "For the matching pixel in HPX_PIX_512, the STAR population's "
        "own unthinned intrinsic count on the common (log10 xi, "
        "log10 F_4.5) grid: the expected count with every member's own "
        "probability of being cataloged set to one (no flux cut, no "
        "dimming), so N_CAT_CELL_STAR never exceeds this cell by cell."),
    ("prior_atlas_hpx512", "N_CELL_AGB"): ("objects",
        "For the matching pixel in HPX_PIX_512, the AGB population's own "
        "unthinned intrinsic count on the common (log10 xi, log10 F_4.5) "
        "grid: the expected count with every member's own probability "
        "of being cataloged set to one (no flux cut, no dimming), so "
        "N_CAT_CELL_AGB never exceeds this cell by cell."),
    ("prior_atlas_hpx512", "N_CELL_PAHC"): ("objects",
        "For the matching pixel in HPX_PIX_512, the PAHC population's "
        "own unthinned intrinsic count on the common (log10 xi, "
        "log10 F_4.5) grid: the expected count with every member's own "
        "probability of being cataloged set to one (no flux cut, no "
        "dimming), so N_CAT_CELL_PAHC never exceeds this cell by cell."),
    ("prior_atlas_hpx512", "N_CELL_GAL"): ("objects",
        "For the matching pixel in HPX_PIX_512, the background-galaxy "
        "population's own unthinned intrinsic count on the common "
        "(log10 xi, log10 F_4.5) grid: the expected count with every "
        "member's own probability of being cataloged set to one (no "
        "flux cut, no dimming), so N_CAT_CELL_GAL never exceeds this "
        "cell by cell."),
    ("prior_atlas_hpx512", "N_CELL_YSO"): ("objects",
        "For the matching pixel in HPX_PIX_512, the YSO population's own "
        "unthinned intrinsic count on the common (log10 xi, log10 F_4.5) "
        "grid: the expected count with every member's own probability "
        "of being cataloged set to one (no flux cut, no dimming), so "
        "N_CAT_CELL_YSO never exceeds this cell by cell."),
    ("prior_atlas_hpx512", "N_CELL_H2S"): ("objects",
        "For the matching pixel in HPX_PIX_512, the H2-shock "
        "population's own unthinned intrinsic count on the common "
        "(log10 xi, log10 F_4.5) grid: the expected count with every "
        "member's own probability of being cataloged set to one (no "
        "flux cut, no dimming), so N_CAT_CELL_H2S never exceeds this "
        "cell by cell."),
    ("prior_atlas_hpx512", "SHARE_STAR"): ("dimensionless",
        "For the matching pixel in HPX_PIX_512, the STAR population's "
        "own share of this pixel's total cataloged density: N_CAT_STAR "
        "divided by the sum of N_CAT_<class> over every class."),
    ("prior_atlas_hpx512", "SHARE_AGB"): ("dimensionless",
        "For the matching pixel in HPX_PIX_512, the AGB population's own "
        "share of this pixel's total cataloged density: N_CAT_AGB "
        "divided by the sum of N_CAT_<class> over every class."),
    ("prior_atlas_hpx512", "SHARE_PAHC"): ("dimensionless",
        "For the matching pixel in HPX_PIX_512, the PAHC population's "
        "own share of this pixel's total cataloged density: N_CAT_PAHC "
        "divided by the sum of N_CAT_<class> over every class."),
    ("prior_atlas_hpx512", "SHARE_GAL"): ("dimensionless",
        "For the matching pixel in HPX_PIX_512, the background-galaxy "
        "population's own share of this pixel's total cataloged "
        "density: N_CAT_GAL divided by the sum of N_CAT_<class> over "
        "every class."),
    ("prior_atlas_hpx512", "SHARE_YSO"): ("dimensionless",
        "For the matching pixel in HPX_PIX_512, the YSO population's own "
        "share of this pixel's total cataloged density: N_CAT_YSO "
        "divided by the sum of N_CAT_<class> over every class."),
    ("prior_atlas_hpx512", "SHARE_H2S"): ("dimensionless",
        "For the matching pixel in HPX_PIX_512, the H2-shock "
        "population's own share of this pixel's total cataloged "
        "density: N_CAT_H2S divided by the sum of N_CAT_<class> over "
        "every class."),
    ("prior_atlas_hpx512", "N_OBS"): ("sources",
        "For the matching pixel in HPX_PIX_512, how many sources the "
        "real SESNA catalog holds there."),
    ("prior_atlas_hpx512", "N_OBS_BRIGHT3"): ("sources",
        "For the matching pixel in HPX_PIX_512, how many sources the "
        "real SESNA catalog holds there that are brighter than three "
        "times this pixel's own I2 50%-completeness limit."),
    ("prior_atlas_hpx512", "N_OBS_BRIGHT10"): ("sources",
        "For the matching pixel in HPX_PIX_512, how many sources the "
        "real SESNA catalog holds there that are brighter than ten "
        "times this pixel's own I2 50%-completeness limit."),

    # bmstp.density -- table_density_source (P1): one row per curated
    # source, its column, grain indices, detection limits and the six sky
    # densities.
    ("table_density_source", "NAME"): ("source name",
        "The source's name from the SESNA catalog, in this file's row "
        "order (the curated catalog's own order)."),
    ("table_density_source", "A_COL_K"): ("mag A_K",
        "This source's own adopted dust-column extinction: the "
        "Herschel arm's value where Herschel covers the source, the "
        "Planck arm's value elsewhere."),
    ("table_density_source", "A_COL_SIG_K"): ("mag A_K",
        "The 1-sigma uncertainty on A_COL_K."),
    ("table_density_source", "ARM"): ("code",
        "Which arm A_COL_K came from: 0 Herschel, 1 Planck."),
    ("table_density_source", "ZP_SIG_K"): ("mag A_K",
        "The uncertainty on the Herschel field zero-point offset already "
        "subtracted from A_COL_K; 0 for a Planck-arm source."),
    ("table_density_source", "TILE"): ("tile index",
        "This source's own row into the per-tile STAR/AGB shape grid: "
        "which tile this source belongs to."),
    ("table_density_source", "SIGHTLINE_ROW"): ("row index",
        "This source's own row into the per-sightline YSO/H2S shape "
        "grid: which sightline this source's position falls on."),
    ("table_density_source", "HPX_512"): ("nested HEALPix pixel, nside 512",
        "This source's own nested nside-512 pixel number."),
    ("table_density_source", "F_LIM_50_MJY"): ("mJy",
        "This source's own 50%-completeness flux limit in each of the "
        "eight bands (J, H, Ks, 3.6, 4.5, 5.8, 8.0 and 24 micron, this "
        "project's own band order), the catalog's per-source limits "
        "product."),
    ("table_density_source", "D_PAHC"): ("dimensionless (log10 mJy)",
        "Minus log10 of this source's own 8 micron completeness-limit "
        "flux (F_LIM_50_MJY's 8 micron column): the PAHC contamination "
        "test's own depth term, before dividing by a predicted "
        "photospheric flux."),
    ("table_density_source", "A_CLOUD_K"): ("mag A_K",
        "This source's own cloud-foreground extinction: A_COL_K times "
        "the sightline's own foreground share of the column (nothing "
        "behind the cloud interval's front edge is deducted, since the "
        "3-D map cannot partition column reliably behind a cloud at a "
        "kiloparsec)."),
    ("table_density_source", "DENSITY_STAR"): ("objects per square degree",
        "This source's own retained STAR sky density: the field-star "
        "population's tile density at this source's own tile."),
    ("table_density_source", "DENSITY_AGB"): ("objects per square degree",
        "This source's own retained AGB sky density: the evolved-star "
        "population's tile density at this source's own tile."),
    ("table_density_source", "DENSITY_PAHC"): ("objects per square degree",
        "This source's own retained PAHC sky density, identical to "
        "DENSITY_STAR (PAHC shares the STAR population's own grid)."),
    ("table_density_source", "DENSITY_GAL"): ("objects per square degree",
        "This source's own retained background-galaxy sky density: the "
        "one survey-wide galaxy counts-law density, the same number for "
        "every source."),
    ("table_density_source", "DENSITY_YSO"): ("objects per square degree",
        "This source's own retained YSO sky density: the young-star law "
        "applied to A_CLOUD_K and this source's own arm, times the "
        "sightline's own on-grid retained fraction."),
    ("table_density_source", "DENSITY_H2S"): ("objects per square degree",
        "This source's own retained H2-shock sky density: the intrinsic "
        "young-star law density (convolved with the region's own HGBS "
        "map for a Herschel-arm source the convolution reaches) times "
        "the region's own knot rate and universal external fraction, "
        "times the sightline's own on-grid retained fraction."),

    # bmstp.template_weights -- P5, six template libraries
    # (<lib>_weights_<granule>): one factor per external population
    # statement the library's own templates are reweighted by, stored per
    # template and brightness cell on the common log10(F_4.5) axis.
    ("yso_weights_region", "MODEL_NAME"): ("source name",
        "This YSO template's own identifier, in this file's row order."),
    ("yso_weights_region", "C_THETA"): ("log10 mJy at 1 kpc",
        "This YSO template's own offset onto the common brightness axis: "
        "its reference 4.5 micron flux, scaled to 1 kpc, as log10."),
    ("yso_weights_region", "LOG10_F45_CENTERS"): ("log10 mJy at 1 kpc",
        "The centers of the 110 common brightness cells every factor "
        "table in this file is tabulated on."),
    ("yso_weights_region", "W"): ("dimensionless",
        "For every YSO template and every cell of LOG10_F45_CENTERS, this "
        "factor's own per-template weight: how the Dunham et al. (2015) "
        "YSO census's brightness density (or, for a template-matched "
        "factor, the region's own shift kernel) reweights that template "
        "relative to the library's own template density, at that "
        "brightness. Which population statement this factor encodes is "
        "named by this dataset's own enclosing group's NAME attribute."),
    ("yso_weights_region", "C_F"): ("dimensionless",
        "For every YSO template, this factor's own per-template offset: "
        "a correction applied to the template's weight at read time, "
        "beyond the per-cell table in W."),

    ("galz_weights_survey", "MODEL_NAME"): ("source name",
        "This galaxy template's own identifier, in this file's row "
        "order."),
    ("galz_weights_survey", "C_THETA"): ("log10 mJy at 1 kpc",
        "This galaxy template's own offset onto the common brightness "
        "axis: its reference 4.5 micron flux, scaled to 1 kpc, as log10."),
    ("galz_weights_survey", "LOG10_F45_CENTERS"): ("log10 mJy at 1 kpc",
        "The centers of the 110 common brightness cells every factor "
        "table in this file is tabulated on."),
    ("galz_weights_survey", "W"): ("dimensionless",
        "For every galaxy template and every cell of LOG10_F45_CENTERS, "
        "this factor's own per-template weight: the galaxy color "
        "Gaussian kernel density reweighting that template relative to "
        "the library's own color density. Which population statement "
        "this factor encodes is named by this dataset's own enclosing "
        "group's NAME attribute."),
    ("galz_weights_survey", "C_F"): ("dimensionless",
        "For every galaxy template, this factor's own per-template "
        "offset: a correction applied to the template's weight at read "
        "time, beyond the per-cell table in W."),

    ("sps_weights_region", "MODEL_NAME"): ("source name",
        "This stellar-atmosphere template's own identifier, in this "
        "file's row order."),
    ("sps_weights_region", "C_THETA"): ("log10 mJy at 1 kpc",
        "This atmosphere template's own offset onto the common brightness "
        "axis: its reference 4.5 micron flux, scaled to 1 kpc, as log10."),
    ("sps_weights_region", "LOG10_F45_CENTERS"): ("log10 mJy at 1 kpc",
        "The centers of the 110 common brightness cells every factor "
        "table in this file is tabulated on."),
    ("sps_weights_region", "W"): ("dimensionless",
        "For every atmosphere template and every cell of "
        "LOG10_F45_CENTERS, this factor's own per-template weight: this "
        "region's own matched-star count for that template's atmosphere "
        "type, needing no division by a library density since the count "
        "is already a weight. Which population statement this factor "
        "encodes is named by this dataset's own enclosing group's NAME "
        "attribute."),
    ("sps_weights_region", "C_F"): ("dimensionless",
        "For every atmosphere template, this factor's own per-template "
        "offset: a correction applied to the template's weight at read "
        "time, beyond the per-cell table in W."),

    ("agb_weights_region", "MODEL_NAME"): ("source name",
        "This AGB template's own identifier, in this file's row order."),
    ("agb_weights_region", "C_THETA"): ("log10 mJy at 1 kpc",
        "This AGB template's own offset onto the common brightness axis: "
        "its reference 4.5 micron flux, scaled to 1 kpc, as log10."),
    ("agb_weights_region", "LOG10_F45_CENTERS"): ("log10 mJy at 1 kpc",
        "The centers of the 110 common brightness cells every factor "
        "table in this file is tabulated on."),
    ("agb_weights_region", "W"): ("dimensionless",
        "For every AGB template and every cell of LOG10_F45_CENTERS, "
        "this factor's own per-template weight: the Riebel et al. (2012) "
        "GRAMS optical-depth census, by dust chemistry, reweighting that "
        "template relative to the library's own optical-depth density. "
        "Which population statement this factor encodes is named by this "
        "dataset's own enclosing group's NAME attribute."),
    ("agb_weights_region", "C_F"): ("dimensionless",
        "For every AGB template, this factor's own per-template offset: "
        "a correction applied to the template's weight at read time, "
        "beyond the per-cell table in W."),

    ("pahc_weights_region", "MODEL_NAME"): ("source name",
        "This PAHC template's own identifier, in this file's row order."),
    ("pahc_weights_region", "C_THETA"): ("log10 mJy at 1 kpc",
        "This PAHC template's own offset onto the common brightness axis: "
        "its reference 4.5 micron flux, scaled to 1 kpc, as log10."),
    ("pahc_weights_region", "LOG10_F45_CENTERS"): ("log10 mJy at 1 kpc",
        "The centers of the 110 common brightness cells every factor "
        "table in this file is tabulated on."),
    ("pahc_weights_region", "W"): ("dimensionless",
        "For every PAHC template and every cell of LOG10_F45_CENTERS, "
        "this factor's own per-template weight: this region's own sps "
        "atmosphere-type count, borrowed at each PAHC template's nearest "
        "atmosphere match since the PAHC library carries no atmosphere-"
        "type axis of its own, needing no further division by a library "
        "density. Which population statement this factor encodes is "
        "named by this dataset's own enclosing group's NAME attribute."),
    ("pahc_weights_region", "C_F"): ("dimensionless",
        "For every PAHC template, this factor's own per-template offset: "
        "a correction applied to the template's weight at read time, "
        "beyond the per-cell table in W."),

    ("h2shock_weights_region", "MODEL_NAME"): ("source name",
        "This h2shock template's own identifier, in this file's row "
        "order."),
    ("h2shock_weights_region", "C_THETA"): ("log10 mJy at 1 kpc",
        "This h2shock template's own offset onto the common brightness "
        "axis, from its own Sigma-to-4.5-micron conversion, applied once "
        "at build time (not reapplied when this column is read)."),
    ("h2shock_weights_region", "LOG10_F45_CENTERS"): ("log10 mJy at 1 kpc",
        "The centers of the 110 common brightness cells every factor "
        "table in this file is tabulated on."),
    ("h2shock_weights_region", "W"): ("dimensionless",
        "For every h2shock template and every cell of LOG10_F45_CENTERS, "
        "this factor's own per-template weight: the uniform template "
        "weight, since h2shock's conditional table already carries the "
        "region's own knot-brightness distribution (no external density "
        "to divide by). Which population statement this factor encodes "
        "is named by this dataset's own enclosing group's NAME attribute."),
    ("h2shock_weights_region", "C_F"): ("dimensionless",
        "For every h2shock template, this factor's own per-template "
        "offset: a correction applied to the template's weight at read "
        "time, beyond the per-cell table in W."),

    # bmstp.shapes -- star_shape_tile (P2, per tile), cloud_shape_sightline
    # (P3, per sightline) and gal_shape_survey (P4, survey-wide): each
    # class's population sample binned onto the common (log10 xi,
    # log10 F_4.5) shape grid.
    ("star_shape_tile", "LOG10_XI_EDGES"): ("log10(dimensionless)",
        "The edges of the log10(depth fraction) axis every shape grid in "
        "this package shares: 10**x is the dimensionless depth fraction "
        "xi, 0 at the observer and 1 at a sightline's own total column."),
    ("star_shape_tile", "LOG10_F45_EDGES"): ("log10 mJy at 1 kpc",
        "The edges of the common log10(4.5 micron flux, scaled to 1 kpc) "
        "axis every shape grid in this package shares: 10**x is a flux "
        "in mJy at the register's 1 kpc reference distance."),
    ("star_shape_tile", "TILE_ID"): ("tile index",
        "The tile each row of GRID_STAR, GRID_AGB and the other per-tile "
        "datasets in this file describes, in this file's own row order."),
    ("star_shape_tile", "GRID_STAR"): ("dimensionless",
        "For the matching tile in TILE_ID, the normal (non-evolved) field "
        "stars' own population density on the common (log10 xi, "
        "log10 F_4.5) grid: integrating over a cell gives that cell's own "
        "share of the tile's STAR population, up to the floor ON_GRID_STAR "
        "leaves out."),
    ("star_shape_tile", "GRID_AGB"): ("dimensionless",
        "For the matching tile in TILE_ID, the evolved (AGB) field stars' "
        "own population density on the common (log10 xi, log10 F_4.5) "
        "grid: integrating over a cell gives that cell's own share of "
        "the tile's AGB population, up to the floor ON_GRID_AGB leaves "
        "out."),
    ("star_shape_tile", "MASS_OUTSIDE_STAR"): ("dimensionless",
        "For the matching tile in TILE_ID, the fraction of this tile's "
        "own STAR population weight that falls outside the common grid "
        "(below the retention limit or beyond an edge), not stored in "
        "GRID_STAR."),
    ("star_shape_tile", "MASS_OUTSIDE_AGB"): ("dimensionless",
        "For the matching tile in TILE_ID, the fraction of this tile's "
        "own AGB population weight that falls outside the common grid, "
        "not stored in GRID_AGB."),
    ("star_shape_tile", "ON_GRID_STAR"): ("dimensionless",
        "1 minus MASS_OUTSIDE_STAR: the fraction of this tile's own STAR "
        "population weight that GRID_STAR actually holds."),
    ("star_shape_tile", "ON_GRID_AGB"): ("dimensionless",
        "1 minus MASS_OUTSIDE_AGB: the fraction of this tile's own AGB "
        "population weight that GRID_AGB actually holds."),

    ("cloud_shape_sightline", "LOG10_XI_EDGES"): ("log10(dimensionless)",
        "The edges of the log10(depth fraction) axis every shape grid in "
        "this package shares: 10**x is the dimensionless depth fraction "
        "xi, 0 at the observer and 1 at a sightline's own total column."),
    ("cloud_shape_sightline", "LOG10_F45_EDGES"): ("log10 mJy at 1 kpc",
        "The edges of the common log10(4.5 micron flux, scaled to 1 kpc) "
        "axis every shape grid in this package shares: 10**x is a flux "
        "in mJy at the register's 1 kpc reference distance."),
    ("cloud_shape_sightline", "HPX_PIX_256"): ("nested HEALPix pixel, nside 256",
        "This region's own occupied nside-256 pixel numbers (nested "
        "ordering). Row i here is row i of GRID_YSO, XI_MARGINAL and the "
        "other per-sightline datasets in this file."),
    ("cloud_shape_sightline", "GRID_YSO"): ("dimensionless",
        "For the matching sightline in HPX_PIX_256, the YSO population "
        "density on the common (log10 xi, log10 F_4.5) grid, restricted "
        "to this region's own cloud interval: integrating over a cell "
        "gives that cell's own share of the sightline's YSO population, "
        "up to the floor ON_GRID_YSO leaves out."),
    ("cloud_shape_sightline", "XI_MARGINAL"): ("dimensionless",
        "For the matching sightline in HPX_PIX_256, GRID_YSO summed over "
        "the brightness axis: the YSO population's own depth-fraction "
        "distribution alone, restricted to this region's cloud interval."),
    ("cloud_shape_sightline", "MASS_OUTSIDE_YSO"): ("dimensionless",
        "For the matching sightline in HPX_PIX_256, the fraction of this "
        "sightline's own YSO population weight that falls outside the "
        "common grid, not stored in GRID_YSO."),
    ("cloud_shape_sightline", "ON_GRID_YSO"): ("dimensionless",
        "1 minus MASS_OUTSIDE_YSO: the fraction of this sightline's own "
        "YSO population weight that GRID_YSO actually holds."),
    ("cloud_shape_sightline", "GRID_H2S"): ("dimensionless",
        "For the matching sightline in HPX_PIX_256, the H2 shock "
        "population density on the common (log10 xi, log10 F_4.5) grid: "
        "XI_MARGINAL times this region's own H2-shock brightness "
        "distribution (this file's LOGSIG_MEAN/LOGSIG_STD attributes "
        "convolved with the h2shock template conversions): integrating "
        "over a cell gives that cell's own share of the sightline's H2 "
        "shock population, up to the floor ON_GRID_H2S leaves out."),
    ("cloud_shape_sightline", "MASS_OUTSIDE_H2S"): ("dimensionless",
        "For the matching sightline in HPX_PIX_256, the fraction of this "
        "sightline's own H2 shock population weight that falls outside "
        "the common grid, not stored in GRID_H2S."),
    ("cloud_shape_sightline", "ON_GRID_H2S"): ("dimensionless",
        "1 minus MASS_OUTSIDE_H2S: the fraction of this sightline's own "
        "H2 shock population weight that GRID_H2S actually holds."),

    ("gal_shape_survey", "LOG10_XI_EDGES"): ("log10(dimensionless)",
        "The edges of the log10(depth fraction) axis every shape grid in "
        "this package shares: 10**x is the dimensionless depth fraction "
        "xi, 0 at the observer and 1 at a sightline's own total column."),
    ("gal_shape_survey", "LOG10_F45_EDGES"): ("log10 mJy at 1 kpc",
        "The edges of the common log10(4.5 micron flux, scaled to 1 kpc) "
        "axis every shape grid in this package shares: 10**x is a flux "
        "in mJy at the register's 1 kpc reference distance."),
    ("gal_shape_survey", "GRID"): ("dimensionless",
        "The survey-wide background-galaxy population density on the "
        "common (log10 xi, log10 F_4.5) grid: a delta at xi = 0 (a "
        "galaxy's light passes through the whole sightline) times the "
        "background-galaxy counts law in F_4.5, up to the floor this "
        "file's ON_GRID_GAL attribute leaves out."),

    # bmstp.cloud_interval -- cloud_interval_shape_sightline: each
    # sightline's own dense fraction between the region's cloud interval,
    # per region.
    ("cloud_interval_shape_sightline", "HPX_PIX_256"): ("nested HEALPix pixel, nside 256",
        "This region's own occupied nside-256 pixel numbers (nested "
        "ordering). Row i here is row i of XI_FRONT, XI_BACK and "
        "W_CLOUD."),
    ("cloud_interval_shape_sightline", "XI_FRONT"): ("dimensionless",
        "For the matching sightline in HPX_PIX_256, its own cumulative "
        "extinction profile u = A(d)/A(inf) evaluated at this region's "
        "cloud interval's own front edge (this file's D_FRONT_PC "
        "attribute)."),
    ("cloud_interval_shape_sightline", "XI_BACK"): ("dimensionless",
        "For the matching sightline in HPX_PIX_256, its own cumulative "
        "extinction profile u = A(d)/A(inf) evaluated at this region's "
        "cloud interval's own back edge (this file's D_BACK_PC "
        "attribute). Never smaller than XI_FRONT."),
    ("cloud_interval_shape_sightline", "W_CLOUD"): ("dimensionless",
        "For the matching sightline in HPX_PIX_256, the diffuse/dense "
        "extinction-law ramp weight evaluated at the column of dust "
        "between the cloud interval's front and back edges, (XI_BACK - "
        "XI_FRONT) times this sightline's own total extinction column. "
        "0 where that column is 0."),

    # population.kernel -- kernel_sesna_survey: the column kernel p(T |
    # A_measured), a two-log-normal mixture per arm and node, survey-wide,
    # plus the cloud-class within-beam tilt fit.
    ("kernel_sesna_survey", "A_NODES"): ("mag A_K",
        "The fixed column ladder (this package's own column-grid product) "
        "the kernel's mixture is tabulated on."),
    ("kernel_sesna_survey", "MIX_W"): ("dimensionless",
        "For each of the two arms (row 0 Herschel, row 1 Planck) and each "
        "node of A_NODES, the mixing weight of the first of the two "
        "log-normal components in the column kernel's mixture."),
    ("kernel_sesna_survey", "MIX_MU"): ("dex",
        "For each of the two arms (index 0 Herschel, index 1 Planck), "
        "each node of A_NODES and each of the two mixture components "
        "(last axis), the component's own log10 offset from the node's "
        "measured column: log10(true column) = log10(A_NODES) + MIX_MU, "
        "recentred so the mixture's own linear-space mean is exactly the "
        "node's measured column."),
    ("kernel_sesna_survey", "MIX_SIGMA"): ("dex",
        "For each of the two arms, each node of A_NODES and each of the "
        "two mixture components (last axis), that component's own "
        "standard deviation in log10(true column), before any per-source "
        "measurement or zero-point uncertainty is added in quadrature."),
    ("kernel_sesna_survey", "ZP_HERSCHEL_K"): ("mag A_K",
        "The survey-wide root-mean-square Herschel field zero-point "
        "offset, used as a source's own ZP_SIGMA_K only where a caller "
        "does not supply that source's own field-specific value."),
    ("kernel_sesna_survey", "CLOUD_SIGMA_HERSCHEL_DEX"): ("dex",
        "The fitted additional structural width, in log10(true column), "
        "of the cloud classes (YSO, H2S) at the Herschel arm's own beam, "
        "from the joint fit on HOPS/eHOPS protostars."),
    ("kernel_sesna_survey", "CLOUD_SIGMA_HERSCHEL_P16"): ("dex",
        "The 16th-percentile bound of CLOUD_SIGMA_HERSCHEL_DEX's own "
        "68% confidence interval from the joint fit."),
    ("kernel_sesna_survey", "CLOUD_SIGMA_HERSCHEL_P84"): ("dex",
        "The 84th-percentile bound of CLOUD_SIGMA_HERSCHEL_DEX's own "
        "68% confidence interval from the joint fit."),
    ("kernel_sesna_survey", "CLOUD_GAMMA_HERSCHEL"): ("dimensionless",
        "The fitted within-beam column exponent for the cloud classes "
        "(YSO, H2S) at the Herschel arm's own beam: 0 if the joint fit's "
        "own 68% interval for this exponent includes 0, the fitted value "
        "otherwise."),
    ("kernel_sesna_survey", "CLOUD_GAMMA_HERSCHEL_P16"): ("dimensionless",
        "The 16th-percentile bound of the fitted exponent's own 68% "
        "confidence interval, before the zero-inclusion rule that sets "
        "CLOUD_GAMMA_HERSCHEL to 0."),
    ("kernel_sesna_survey", "CLOUD_GAMMA_HERSCHEL_P84"): ("dimensionless",
        "The 84th-percentile bound of the fitted exponent's own 68% "
        "confidence interval, before the zero-inclusion rule that sets "
        "CLOUD_GAMMA_HERSCHEL to 0."),
    ("kernel_sesna_survey", "CLOUD_SIGMA_GRID_DEX"): ("dex",
        "The grid of structural-width values the joint maximum-likelihood "
        "fit searched over to find CLOUD_SIGMA_HERSCHEL_DEX."),
    ("kernel_sesna_survey", "CLOUD_GAMMA_GRID"): ("dimensionless",
        "The grid of within-beam exponent values the joint maximum-"
        "likelihood fit searched over to find CLOUD_GAMMA_HERSCHEL."),
    ("kernel_sesna_survey", "CLOUD_GAMMA_SIGMA_LOGLIKE"): ("nats",
        "The joint fit's own log-likelihood surface over "
        "CLOUD_GAMMA_GRID (rows) and CLOUD_SIGMA_GRID_DEX (columns), as a "
        "natural logarithm."),
    ("kernel_sesna_survey", "N_PROTOSTARS_FIT"): ("protostars",
        "How many HOPS/eHOPS protostars (Orion A, Aquila) went into the "
        "joint CLOUD_GAMMA_HERSCHEL/CLOUD_SIGMA_HERSCHEL_DEX fit."),

    # population.star_population -- population_star_tile: per-tile field-
    # star placement, anchor weight and brightness unit, per region (the
    # same dataset names repeat in every tile_<id> group this file holds).
    ("population_star_tile", "LIMIT8_GRID_MJY"): ("mJy",
        "The region's own 8 micron completeness-limit grid: the 5th, "
        "20th, 35th, 50th, 65th, 80th, 95th and 99th percentiles of this "
        "region's sources' own 50%-completeness limit at 8 micron, the "
        "nodes P_PAHC is tabulated on."),
    ("population_star_tile", "STAR_INDEX"): ("row index",
        "For every star this tile draws, its own row in this region's "
        "field-star product's retained group."),
    ("population_star_tile", "XI"): ("dimensionless",
        "For every star this tile draws, its placement fraction along "
        "this tile's own mean extinction profile: u = A(d)/A(inf) at the "
        "star's own distance. A consumer forms the star's own extinction "
        "as a real source's adopted column times this fraction."),
    ("population_star_tile", "W"): ("dimensionless",
        "For every star this tile draws, its total anchor reweighting "
        "factor: the STAR anchor's own Gaia/2MASS joint or marginal "
        "weight at the star's own placement and predicted magnitude. "
        "W_STAR plus W_AGB equals this value exactly, row by row."),
    ("population_star_tile", "W_STAR"): ("dimensionless",
        "The non-evolved share of W for every star this tile draws: W "
        "itself for a star that is not evolved, 0 for one that is."),
    ("population_star_tile", "W_AGB"): ("dimensionless",
        "The evolved (AGB) share of W for every star this tile draws: "
        "the region's own dust-production fraction F_DUSTY_MEAN times W "
        "for a star flagged IS_EVOLVED, 0 otherwise."),
    ("population_star_tile", "IS_EVOLVED"): ("boolean (0/1)",
        "For every star this tile draws, whether TRILEGAL's own raw "
        "surface gravity, temperature and luminosity place it past the "
        "evolved-star HR-diagram cut (1) or not (0)."),
    ("population_star_tile", "LOG10_B"): ("dex",
        "For every star this tile draws, its STAR brightness unit: the "
        "median, over the eight bands, of its own intrinsic TRILEGAL "
        "flux divided by its matched atmosphere template's own reference "
        "flux, as log10."),
    ("population_star_tile", "LOG10_B_PAHC"): ("dex",
        "For every star this tile draws, its PAHC brightness unit: the "
        "median, over the J, H and Ks bands only, of its own intrinsic "
        "TRILEGAL flux divided by the PAHC library's own continuum "
        "reference flux at its matched atmosphere template, as log10."),
    ("population_star_tile", "LOG10_B_AGB_C"): ("dex",
        "For every star this tile draws, the AGB brightness unit under "
        "the carbon-rich dust chemistry, as log10. NaN for a star with "
        "IS_EVOLVED False."),
    ("population_star_tile", "LOG10_B_AGB_O"): ("dex",
        "For every star this tile draws, the AGB brightness unit under "
        "the oxygen-rich dust chemistry, as log10. NaN for a star with "
        "IS_EVOLVED False."),
    ("population_star_tile", "P_PAHC"): ("dimensionless",
        "For every star this tile draws and each of the eight nodes of "
        "LIMIT8_GRID_MJY, the probability that its aperture is "
        "contaminated by PAH nebular emission at that assumed 8 micron "
        "completeness limit, from the PAH-contamination probability "
        "curve P(q) read at this star's own tile-dimmed 8 micron flux."),

    # population.field_stars -- field-stars_trilegal_region: the synthetic
    # TRILEGAL field-star population, retained sample and pre-retention
    # raw columns, per region.
    ("field-stars_trilegal_region", "DIST_PC"): ("pc",
        "This retained synthetic star's true distance from the Sun, "
        "TRILEGAL's own simulated value."),
    ("field-stars_trilegal_region", "LOG_TEFF"): ("log10 K",
        "This retained synthetic star's effective temperature: 10**x is "
        "a temperature in Kelvin."),
    ("field-stars_trilegal_region", "LOG_G"): ("log10(cm/s^2)",
        "This retained synthetic star's surface gravity: 10**x is a "
        "gravity in cm/s^2."),
    ("field-stars_trilegal_region", "LOG_L"): ("log10 Lsun",
        "This retained synthetic star's bolometric luminosity: 10**x is "
        "a luminosity in solar luminosities."),
    ("field-stars_trilegal_region", "FNU_MJY"): ("mJy",
        "This retained synthetic star's intrinsic (undimmed) flux "
        "density in each of the eight bands (J, H, Ks, 3.6, 4.5, 5.8, 8.0 "
        "and 24 micron, this file's own band order), TRILEGAL's own "
        "simulated 2MASS+Spitzer output."),
    ("field-stars_trilegal_region", "G_PROXY"): ("mag Gaia G",
        "This retained synthetic star's intrinsic Gaia G magnitude "
        "proxy: its Ks magnitude plus TRILEGAL's own Gaia G minus 2MASS "
        "Ks color relation, read at its own atmosphere, or, where no "
        "populated cell of that relation is within one step, its "
        "matched atmosphere template's own G-Ks color."),
    ("field-stars_trilegal_region", "KS_MAG"): ("mag 2MASS Ks",
        "This retained synthetic star's intrinsic 2MASS Ks magnitude, "
        "TRILEGAL's own simulated value."),
    ("field-stars_trilegal_region", "K_G_DIFFUSE"): ("A_G/A_V",
        "This retained synthetic star's Gaia extinction coefficient "
        "under the diffuse-ISM law (Danielski et al. 2018), from its "
        "matched atmosphere template."),
    ("field-stars_trilegal_region", "K_G_DENSE"): ("A_G/A_V",
        "This retained synthetic star's Gaia extinction coefficient "
        "under the dense-cloud law (Danielski et al. 2018), from its "
        "matched atmosphere template."),
    ("field-stars_trilegal_region", "TEMPLATE_INDEX"): ("template index",
        "This retained synthetic star's nearest atmosphere template, as "
        "a row position in the project's stellar-atmosphere template "
        "library."),
    ("field-stars_trilegal_region", "POINTING_INDEX"): ("pointing index",
        "This retained synthetic star's own TRILEGAL pointing, as a "
        "position in this region's list of simulated sky pointings."),
    ("field-stars_trilegal_region", "RAW/G_PROXY"): ("mag Gaia G",
        "For every simulated star before the retention cut, its "
        "intrinsic Gaia G magnitude proxy: its Ks magnitude plus the "
        "TRILEGAL G-Ks color relation read at its own atmosphere, or, "
        "where no populated cell of that relation is within one step, "
        "its matched atmosphere template's own G-Ks color. Carried for "
        "every simulated star, not only the ones that survive retention, "
        "since predicting the anchors' observed counts needs the whole "
        "simulated population."),
    ("field-stars_trilegal_region", "RAW/KS_MAG"): ("mag 2MASS Ks",
        "For every simulated star before the retention cut, its "
        "intrinsic 2MASS Ks magnitude, TRILEGAL's own simulated value."),
    ("field-stars_trilegal_region", "RAW/DIST_PC"): ("pc",
        "For every simulated star before the retention cut, its true "
        "distance from the Sun, TRILEGAL's own simulated value."),
    ("field-stars_trilegal_region", "RAW/K_G_DIFFUSE"): ("A_G/A_V",
        "For every simulated star before the retention cut, its Gaia "
        "extinction coefficient under the diffuse-ISM law (Danielski "
        "et al. 2018), from its matched atmosphere template."),
    ("field-stars_trilegal_region", "RAW/K_G_DENSE"): ("A_G/A_V",
        "For every simulated star before the retention cut, its Gaia "
        "extinction coefficient under the dense-cloud law (Danielski "
        "et al. 2018), from its matched atmosphere template."),
    ("field-stars_trilegal_region", "RAW/POINTING_INDEX"): ("pointing index",
        "For every simulated star before the retention cut, its own "
        "TRILEGAL pointing, as a position in this region's list of "
        "simulated sky pointings."),

    # population.anchor_weights -- weights_anchors_tile: the STAR
    # per-tile-and-bin anchor reweighting factor W, per region.
    ("weights_anchors_tile", "G_EDGES"): ("mag Gaia G",
        "The edges of the Gaia G magnitude bins W_G and W_REGION_G are "
        "tabulated on."),
    ("weights_anchors_tile", "KS_EDGES"): ("mag 2MASS/UKIDSS Ks",
        "The edges of the Ks magnitude bins W_KS and W_REGION_KS are "
        "tabulated on."),
    ("weights_anchors_tile", "W_G"): ("dimensionless",
        "For each tile and each Gaia G bin of G_EDGES, the anchor "
        "reweighting factor: the sky's own observed star count over the "
        "model's predicted count, shrunk toward the region-pooled value "
        "in the log. Where this tile carries no usable evidence in this "
        "bin, this is W_REGION_G's own value for that bin."),
    ("weights_anchors_tile", "W_REGION_G"): ("dimensionless",
        "One value per Gaia G bin of G_EDGES (shared by every tile): the "
        "region-pooled reweighting factor, or, where no tile in this "
        "region has usable evidence in that bin, the survey-pooled "
        "value, or NaN where neither this region nor the survey has any."),
    ("weights_anchors_tile", "POPULATED_G"): ("boolean",
        "One value per Gaia G bin of G_EDGES: True where at least one "
        "tile of this region carries usable evidence in that bin, so "
        "W_REGION_G is this region's own fit rather than a survey-pooled "
        "or absent value."),
    ("weights_anchors_tile", "W_KS"): ("dimensionless",
        "For each tile and each Ks bin of KS_EDGES, the anchor "
        "reweighting factor on the already completeness-corrected (P_KS) "
        "predicted counts: the sky's own observed star count over the "
        "model's predicted count, shrunk toward the region-pooled value "
        "in the log. Where this tile carries no usable evidence in this "
        "bin, this is W_REGION_KS's own value for that bin."),
    ("weights_anchors_tile", "W_REGION_KS"): ("dimensionless",
        "One value per Ks bin of KS_EDGES (shared by every tile): the "
        "region-pooled reweighting factor, or, where no tile in this "
        "region has usable evidence in that bin, the survey-pooled "
        "value, or NaN where neither this region nor the survey has any."),
    ("weights_anchors_tile", "POPULATED_KS"): ("boolean",
        "One value per Ks bin of KS_EDGES: True where at least one tile "
        "of this region carries usable evidence in that bin, so "
        "W_REGION_KS is this region's own fit rather than a survey-pooled "
        "or absent value."),
    ("weights_anchors_tile", "MEASURED_KS"): ("boolean",
        "For each tile and each Ks bin of KS_EDGES, True where the "
        "model predicts at least one star there: False marks a bin past "
        "this tile's own survey depth (the deep UKIDSS bins where the "
        "region carries no deep coverage), which a consumer must not "
        "read a weight for at all."),
    ("weights_anchors_tile", "W_JOINT"): ("dimensionless",
        "For each tile, each Gaia G bin of G_EDGES and each Ks bin of "
        "KS_EDGES, the anchor reweighting factor on the joint (G, Ks) "
        "grid: the sky's own observed star count over the model's "
        "predicted count in that joint cell, shrunk toward the region-"
        "pooled value in the log. Read only where USE_JOINT is True for "
        "that tile and bin; a star without a usable joint cell takes "
        "the marginal weight instead."),
    ("weights_anchors_tile", "USE_JOINT"): ("boolean",
        "For each tile, each Gaia G bin of G_EDGES and each Ks bin of "
        "KS_EDGES, True where this region's pooled count in that joint "
        "cell reached the fitting floor and W_JOINT is therefore an "
        "actual fit rather than an unfitted placeholder, and the "
        "matching joint observed count for this tile is positive."),
    ("weights_anchors_tile", "W_REGION_JOINT"): ("dimensionless",
        "One value per joint (G, Ks) cell (shared by every tile): the "
        "region-pooled joint reweighting factor, or, where no tile in "
        "this region has usable evidence in that cell, the survey-"
        "pooled value, or NaN where neither this region nor the survey "
        "has any."),
    ("weights_anchors_tile", "EXCLUDED"): ("boolean",
        "One value per tile: True where this tile's own pooled observed-"
        "over-predicted ratio departs from the region's median by more "
        "than the cluster-exclusion band, so its evidence is pooled into "
        "the region fit rather than kept as its own."),
    ("weights_anchors_tile", "P_KS"): ("dimensionless",
        "For each tile and each Ks bin of KS_EDGES, the 2MASS anchor's "
        "own per-tile completeness sigmoid: 1 on every UKIDSS-served bin, "
        "and on a 2MASS-served bin the fitted probability of detection at "
        "that bin's own magnitude given this tile's KS_M50 and this "
        "file's fixed KS_SCALE."),
    ("weights_anchors_tile", "KS_M50"): ("mag 2MASS Ks",
        "One value per tile: the Ks magnitude at which this tile's own "
        "(or, where too few bins clear the counting floor, the region-"
        "summed) completeness sigmoid reaches 50%."),
    ("weights_anchors_tile", "KS_SCALE"): ("mag",
        "The fixed roll-off width of the 2MASS completeness sigmoid P_KS "
        "is evaluated with, the same value for every tile in this file."),

    # population.anchor_tiles -- tiles_anchors_hpx512 and
    # histograms_anchors_hpx512: the STAR anchor tiling and the two
    # anchors' observed/predicted magnitude histograms, per region.
    ("tiles_anchors_hpx512", "HPX_PIX_512"): ("nested HEALPix pixel, nside 512",
        "This region's own occupied nside-512 pixel numbers (nested "
        "ordering). Row i here is row i of TILE_ID."),
    ("tiles_anchors_hpx512", "TILE_ID"): ("tile index",
        "For the matching pixel in HPX_PIX_512, which tile it belongs to, "
        "as a position in TILE_L_DEG, TILE_B_DEG and TILE_OMEGA_DEG2 "
        "(a separate, shorter list in this same file, one row per tile "
        "rather than one row per pixel)."),
    ("tiles_anchors_hpx512", "TILE_L_DEG"): ("deg",
        "One row per tile (not one row per pixel): the tile's own center "
        "Galactic longitude."),
    ("tiles_anchors_hpx512", "TILE_B_DEG"): ("deg",
        "One row per tile: the tile's own center Galactic latitude."),
    ("tiles_anchors_hpx512", "TILE_OMEGA_DEG2"): ("deg^2",
        "One row per tile: the tile's own total solid angle, the sum of "
        "its member pixels' areas."),

    ("histograms_anchors_hpx512", "HPX_PIX_512"): ("nested HEALPix pixel, nside 512",
        "This region's own occupied nside-512 pixel numbers (nested "
        "ordering). Row i here is row i of every other per-pixel dataset "
        "in this file."),
    ("histograms_anchors_hpx512", "A_PIX_K"): ("mag A_K",
        "For the matching pixel in HPX_PIX_512, the extinction column at "
        "that pixel's position, used to deredden the nearest TRILEGAL "
        "pointing's raw stars into this pixel's own predicted counts."),
    ("histograms_anchors_hpx512", "OMEGA_PIX_DEG2"): ("deg^2",
        "For the matching pixel in HPX_PIX_512, the solid angle of one "
        "nside-512 pixel: the same value repeated for every pixel in "
        "this file."),
    ("histograms_anchors_hpx512", "G_EDGES"): ("mag Gaia G",
        "The edges of the Gaia G magnitude bins N_G_OBS, N_G_PRED and "
        "N_GK_PRED are tabulated on."),
    ("histograms_anchors_hpx512", "KS_EDGES"): ("mag 2MASS/UKIDSS Ks",
        "The edges of the Ks magnitude bins N_KS_OBS, N_KS_PRED and "
        "N_GK_PRED are tabulated on, on the shared 2MASS/UKIDSS scale "
        "KS_SOURCE names per bin."),
    ("histograms_anchors_hpx512", "N_G_OBS"): ("stars",
        "For the matching pixel in HPX_PIX_512 and each Gaia G bin of "
        "G_EDGES, the observed Gaia anchor star count, from the external "
        "Gaia anchor-count product."),
    ("histograms_anchors_hpx512", "N_G_PRED"): ("stars",
        "For the matching pixel in HPX_PIX_512 and each Gaia G bin of "
        "G_EDGES, the model's predicted Gaia anchor star count: the "
        "nearest TRILEGAL pointing's raw stars, dereddened by A_PIX_K and "
        "weighted by the Gaia detection probability, divided by that "
        "pointing's own solid angle."),
    ("histograms_anchors_hpx512", "N_KS_OBS"): ("stars",
        "For the matching pixel in HPX_PIX_512 and each Ks bin of "
        "KS_EDGES, the observed 2MASS/UKIDSS anchor star count, from the "
        "external anchor-count products KS_SOURCE names per bin."),
    ("histograms_anchors_hpx512", "N_KS_PRED"): ("stars",
        "For the matching pixel in HPX_PIX_512 and each Ks bin of "
        "KS_EDGES, the model's predicted anchor star count: the nearest "
        "TRILEGAL pointing's raw stars, dereddened by A_PIX_K, divided by "
        "that pointing's own solid angle."),
    ("histograms_anchors_hpx512", "N_GK_PRED"): ("stars",
        "For the matching pixel in HPX_PIX_512, each Gaia G bin of "
        "G_EDGES and each Ks bin of KS_EDGES, the model's predicted "
        "joint count: the nearest TRILEGAL pointing's raw stars, "
        "dereddened by A_PIX_K and weighted by the Gaia detection "
        "probability, divided by that pointing's own solid angle and "
        "binned jointly in (G, Ks) rather than in each magnitude "
        "marginally."),
    ("histograms_anchors_hpx512", "DEEP_COVERED"): ("boolean",
        "For the matching pixel in HPX_PIX_512, True where the region's "
        "deeper UKIDSS GPS survey reaches that pixel, so KS_EDGES runs "
        "past the 2MASS-only cut and KS_SOURCE marks the UKIDSS-only "
        "bins for it."),
    ("histograms_anchors_hpx512", "KS_SOURCE"): ("code",
        "For each Ks bin of KS_EDGES (one value per bin, shared by every "
        "pixel), which survey that bin's counts come from: 0 2MASS, 1 "
        "UKIDSS. Every bin past the 2MASS cut is 1; a region with no "
        "UKIDSS coverage at all has every bin 0."),

    # population.yso_mass -- mass_yso_survey: each pooled YSO register
    # template's own stellar mass, read off a pre-main-sequence track.
    ("mass_yso_survey", "MODEL_NAME"): ("source name",
        "This template's own identifier, in the pooled YSO register's own "
        "row order."),
    ("mass_yso_survey", "M_STAR"): ("Msun",
        "This template's stellar mass: the mass at which the BHAC15 1 Myr "
        "track (or, above its 1.4 solar-mass top, the MIST v1.2 1 Myr "
        "isochrone) has the template's own luminosity LOG10_L, floored at "
        "0.1 solar masses (FLAG_BELOW_FLOOR) and capped at the MIST "
        "track's own top (FLAG_ABOVE_TOP)."),
    ("mass_yso_survey", "LOG10_L"): ("log10 Lsun",
        "This template's own bolometric luminosity, from its radiative-"
        "transfer grid, independent of any stellar-evolution track: "
        "10**x is a luminosity in solar luminosities."),
    ("mass_yso_survey", "T_EFF"): ("K",
        "This template's own effective temperature, from its radiative-"
        "transfer grid, stored for reference but not used to derive "
        "M_STAR."),
    ("mass_yso_survey", "SUBGRID"): ("subclass code",
        "Which of the five YSO sub-grids this template belongs to: C0, "
        "CI, CII, CIII or TD."),
    ("mass_yso_survey", "FLAG_ABOVE_TOP"): ("boolean",
        "True where this template's luminosity exceeds the MIST v1.2 1 "
        "Myr isochrone's own top, so M_STAR is capped at that track's "
        "highest tabulated mass rather than interpolated."),
    ("mass_yso_survey", "FLAG_BELOW_FLOOR"): ("boolean",
        "True where this template's luminosity implies a mass below the "
        "Chabrier (2003) system IMF's own 0.1 solar-mass floor, so "
        "M_STAR is set to that floor."),
    ("mass_yso_survey", "FLAG_HIGH"): ("boolean",
        "True where this template's mass was read from the MIST v1.2 1 "
        "Myr isochrone (above the BHAC15 track's 1.4 solar-mass top) "
        "rather than from the BHAC15 1 Myr track."),

    # population.gal -- counts_gal_survey: the background-galaxy number-
    # counts law, survey-wide.
    ("counts_gal_survey", "LOG10_A"): ("log10(deg^-2)",
        "The fitted broken power law's own normalization: log10 of the "
        "cumulative galaxy count per square degree, N(>S), at the break "
        "flux LOG10_S_BREAK."),
    ("counts_gal_survey", "LOG10_S_BREAK"): ("log10 mJy",
        "The fitted broken power law's own break flux at 4.5 micron: "
        "10**x is a flux in mJy."),
    ("counts_gal_survey", "ALPHA_FAINT"): ("dimensionless",
        "The fitted broken power law's own faint-end slope of log10 N(>S) "
        "against log10 S."),
    ("counts_gal_survey", "ALPHA_BRIGHT"): ("dimensionless",
        "The fitted broken power law's own bright-end slope of log10 "
        "N(>S) against log10 S."),
    ("counts_gal_survey", "SMOOTHNESS"): ("dex",
        "The fitted broken power law's own smoothness parameter: how "
        "many dex in flux the transition between ALPHA_FAINT and "
        "ALPHA_BRIGHT spans around LOG10_S_BREAK."),
    ("counts_gal_survey", "LOG10_S_GRID"): ("log10 mJy",
        "The 4.5 micron flux grid PHI_S, P_POINT and PHI_S_POINT are "
        "tabulated on: 10**x is a flux in mJy."),
    ("counts_gal_survey", "PHI_S"): ("deg^-2 mJy^-1",
        "Fazio et al. (2004)'s intrinsic galaxy differential number-"
        "counts law, phi(S) = -dN/dS, at each flux in LOG10_S_GRID: the "
        "number of galaxies per square degree per mJy of flux at 4.5 "
        "micron."),
    ("counts_gal_survey", "P_POINT"): ("dimensionless",
        "The fraction of galaxies at each flux in LOG10_S_GRID retained "
        "as IRAC point sources rather than resolved and excluded, "
        "measured against Fazio et al. (2004)'s own star counts."),
    ("counts_gal_survey", "PHI_S_POINT"): ("deg^-2 mJy^-1",
        "PHI_S times P_POINT: the point-source-corrected galaxy counts "
        "law every consumer of this product reads."),
    ("counts_gal_survey", "COSMIC_VARIANCE_DEX"): ("dex",
        "The root-mean-square, in dex, of the counts-law fit's own "
        "field-to-field scatter over its candidate fitting variants: the "
        "cosmic-variance uncertainty on PHI_S and PHI_S_POINT."),
    ("counts_gal_survey", "FIT_LOG10_S_MIN"): ("log10 mJy",
        "The faintest flux Fazio et al. (2004)'s own tabulated bins "
        "reach, the lower bound of the range the broken power law was "
        "actually fitted over: 10**x is a flux in mJy."),
    ("counts_gal_survey", "FIT_LOG10_S_MAX"): ("log10 mJy",
        "The brightest flux Fazio et al. (2004)'s own tabulated bins "
        "reach, the upper bound of the range the broken power law was "
        "actually fitted over: 10**x is a flux in mJy."),
    ("counts_gal_survey", "FIT_RMS_DEX"): ("dex",
        "The root-mean-square residual, in dex, of the fitted broken "
        "power law against Fazio et al. (2004)'s own tabulated counts."),

    # population.anchor_observed -- observed_anchors_hpx512: the STAR
    # anchors' observed joint histogram with the model's own young-star
    # and cluster-member counts already subtracted.
    ("observed_anchors_hpx512", "HPX_PIX_512"): ("nested HEALPix pixel, nside 512",
        "This region's own occupied nside-512 anchor pixel numbers "
        "(nested ordering). Row i here is row i of N_G_SUB, N_KS_SUB "
        "and N_GK_SUB."),
    ("observed_anchors_hpx512", "G_EDGES"): ("mag Gaia G",
        "The edges of the Gaia G magnitude bins N_G_SUB and N_GK_SUB are "
        "tabulated on."),
    ("observed_anchors_hpx512", "KS_EDGES"): ("mag 2MASS Ks",
        "The edges of the 2MASS Ks magnitude bins N_KS_SUB and N_GK_SUB "
        "are tabulated on."),
    ("observed_anchors_hpx512", "N_G_SUB"): ("stars",
        "For the matching pixel in HPX_PIX_512 and each Gaia G bin of "
        "G_EDGES, the observed star count with the model's own expected "
        "young-star count and any overlapping bound star cluster's own "
        "member count both subtracted, floored at zero."),
    ("observed_anchors_hpx512", "N_KS_SUB"): ("stars",
        "For the matching pixel in HPX_PIX_512 and each 2MASS Ks bin of "
        "KS_EDGES, the observed star count with the model's own expected "
        "young-star count and any overlapping bound star cluster's own "
        "member count both subtracted, floored at zero."),
    ("observed_anchors_hpx512", "N_GK_SUB"): ("stars",
        "For the matching pixel in HPX_PIX_512, each Gaia G bin of "
        "G_EDGES and each 2MASS Ks bin of KS_EDGES, the Gaia-2MASS joint "
        "crossmatch count with the model's own expected young-star joint "
        "count and any overlapping bound star cluster's own member joint "
        "count both subtracted, floored at zero."),

    # population.yso_law -- law_yso_region: the young-star law's
    # coefficient, fitted per region on the Dunham et al. (2015) census.
    ("law_yso_region", "REGION"): ("region name",
        "The name of the region this row describes, in the fixed thirty-"
        "region order every region-axis product in this package shares."),
    ("law_yso_region", "D_R_PC"): ("pc",
        "This region's own canonical distance, carried over from this "
        "package's region table."),
    ("law_yso_region", "PC2_PER_DEG2"): ("pc^2 per deg^2",
        "The area conversion factor at this region's own distance: one "
        "square degree on the sky subtends this many square parsecs "
        "there."),
    ("law_yso_region", "KAPPA_REGION"): ("young stars pc^-2 per mag^2 of A_K",
        "This region's own fitted young-star law coefficient: the Poisson "
        "maximum-likelihood value matching the Dunham et al. (2015) "
        "census count in this region's fit footprint to the model's "
        "predicted count there. NaN where the footprint holds fewer than "
        "50 census objects."),
    ("law_yso_region", "KAPPA_REGION_LO"): ("young stars pc^-2 per mag^2 of A_K",
        "The lower end of this region's fitted young-star law "
        "coefficient's own 68% Poisson (Garwood) confidence interval. "
        "NaN where this region's fit footprint holds fewer than 50 "
        "Dunham et al. (2015) census objects."),
    ("law_yso_region", "KAPPA_REGION_HI"): ("young stars pc^-2 per mag^2 of A_K",
        "The upper end of this region's fitted young-star law "
        "coefficient's own 68% Poisson (Garwood) confidence interval. "
        "NaN where this region's fit footprint holds fewer than 50 "
        "Dunham et al. (2015) census objects."),
    ("law_yso_region", "N_CENSUS_REGION"): ("YSOs",
        "How many Dunham et al. (2015) census objects, of any class, fall "
        "inside this region's own fit footprint."),
    ("law_yso_region", "FIT_A_K_MIN"): ("mag A_K",
        "The gas-column floor a pixel must clear to enter this region's "
        "own fit footprint: 0.22 mag where the census cloud(s) overlapping "
        "the region are c2d clouds, 0.33 mag where they are Gould Belt "
        "clouds."),
    ("law_yso_region", "KAPPA_USED"): ("young stars pc^-2 per mag^2 of A_K",
        "The young-star law coefficient every other product in this "
        "package reads for this region: KAPPA_REGION where it was fitted, "
        "KAPPA_POOLED otherwise."),
    ("law_yso_region", "KAPPA_POOLED"): ("young stars pc^-2 per mag^2 of A_K",
        "The geometric mean of KAPPA_REGION over every region with a "
        "fitted value, in this file as a whole: the coefficient used for "
        "a region whose own fit did not clear the 50-object minimum."),
    ("law_yso_region", "LAW_BAND_DEX"): ("dex",
        "The root-mean-square of log10(KAPPA_REGION) about "
        "log10(KAPPA_POOLED), over every region with a fitted value: the "
        "cloud-to-cloud scatter of the young-star law coefficient."),
    ("law_yso_region", "CLASS_SHARE_PROTO"): ("dimensionless",
        "The fraction of the whole Dunham et al. (2015) census (every "
        "region) with an infrared spectral index alpha0 >= -0.3 "
        "(protostellar), independent of region."),
    ("law_yso_region", "CLASS_SHARE_DISK"): ("dimensionless",
        "The fraction of the whole Dunham et al. (2015) census with "
        "-1.6 <= alpha0 < -0.3 (disk-bearing), independent of region."),
    ("law_yso_region", "CLASS_SHARE_WEAK"): ("dimensionless",
        "The fraction of the whole Dunham et al. (2015) census with "
        "alpha0 < -1.6 (weak or no disk), independent of region."),

    # population.pahc_curve -- curve_pahc_survey: the PAH-contamination
    # probability curve P(q), survey-wide.
    ("curve_pahc_survey", "LOG10_Q_EDGES"): ("log10(dimensionless)",
        "The edges of the 40 bins in log10(q) this curve is tabulated on, "
        "q being a source's own 8 micron completeness limit divided by "
        "its predicted photospheric 8 micron flux: bin j runs from "
        "LOG10_Q_EDGES[j] to LOG10_Q_EDGES[j+1]."),
    ("curve_pahc_survey", "P_Q"): ("dimensionless",
        "For each bin of LOG10_Q_EDGES, the probability that a source at "
        "that q is contaminated by extended PAH nebular emission in its "
        "aperture: the measured excess fraction, scaled by the bin's own "
        "8-micron-measured share and with the survey's noise rate "
        "subtracted, floored at zero. Zero for every bin whose q is too "
        "small for a 3-sigma nebular excess to register at all (see "
        "P_Q_BRIGHT_EXCESS)."),
    ("curve_pahc_survey", "N_PER_BIN"): ("sources",
        "For each bin of LOG10_Q_EDGES, how many sources with measured "
        "3.6 and 4.5 micron photometry and no 4.5 micron excess fall in "
        "it: this curve's own denominator."),
    ("curve_pahc_survey", "M_PER_BIN"): ("sources",
        "For each bin of LOG10_Q_EDGES, how many of the sources with "
        "measured 3.6 and 4.5 micron photometry and no 4.5 micron "
        "excess that fall in it also have a measured 8 micron flux."),
    ("curve_pahc_survey", "P_Q_BRIGHT_EXCESS"): ("dimensionless",
        "For each bin of LOG10_Q_EDGES whose q is too small for a "
        "3-sigma nebular excess to register (where P_Q is zeroed for "
        "that reason), the measured, unscaled fraction of that bin's "
        "8-micron-measured sources that show an excess: not "
        "contamination but circumstellar 8 micron emission. Zero for "
        "every other bin."),
    ("curve_pahc_survey", "N_PER_BIN_BRIGHT_EXCESS"): ("sources",
        "For each bin of LOG10_Q_EDGES whose q is too small for a "
        "3-sigma nebular excess to register, how many sources with "
        "measured 3.6 and 4.5 micron photometry and no 4.5 micron "
        "excess fall in it. Zero for every other bin."),

    # population.column_grid -- column-grid_sesna_survey: the fixed column
    # ladder every class tabulates its column kernel on.
    ("column-grid_sesna_survey", "A_NODES"): ("mag A_K",
        "The fixed, evenly-log10-spaced column ladder every class's "
        "column kernel is tabulated on, from its floor to its cap, 0.02 "
        "dex per step."),

    # sky.derived.column -- the adopted (gas) column and the extinction
    # (star-light) column, each at source and sightline granule, plus the
    # Herschel/Planck disagreement check.
    ("column_adopted_source", "A_COL_K"): ("mag A_K",
        "For every cataloged source of this region, the adopted dust-"
        "column extinction: the Herschel arm's own A_K, its field's own "
        "zero-point offset subtracted, where Herschel covers the source "
        "and that value is finite and positive; the Planck arm's A_K "
        "otherwise."),
    ("column_adopted_source", "A_COL_SIG_K"): ("mag A_K",
        "The 1-sigma uncertainty on A_COL_K, rebuilt from whichever arm's "
        "own value was adopted: the Herschel arm's random term combined "
        "with ZP_SIGMA_K in quadrature where Herschel was used, the "
        "Planck arm's own uncertainty otherwise."),
    ("column_adopted_source", "A_COL_PROVENANCE"): ("code",
        "Which arm A_COL_K for this source came from: 0 Herschel, 1 "
        "Planck."),
    ("column_adopted_source", "A_COL_FWHM_ARCSEC"): ("arcsec",
        "The beam size of whichever map A_COL_K for this source came "
        "from: the Herschel Gould Belt Survey's 36.3 arcsec where "
        "A_COL_PROVENANCE is 0, the measured Planck beam otherwise."),
    ("column_adopted_source", "HERSCHEL_MAP_ID"): ("map index",
        "For every cataloged source of this region, which map in this "
        "file's own MAP_NAME list covers its position, as a position in "
        "that list. -1 where A_COL_PROVENANCE is 1 (Planck)."),
    ("column_adopted_source", "MAP_NAME"): ("map name",
        "The file name of each HGBS column-density map that reaches some "
        "part of this region, in the order HERSCHEL_MAP_ID indexes."),
    ("column_adopted_source", "ZP_SIGMA_K"): ("mag A_K",
        "For every cataloged source of this region, the uncertainty on "
        "the Herschel field zero-point offset already subtracted from "
        "A_COL_K: 0 where A_COL_PROVENANCE is 1 (Planck), or where the "
        "source's own Herschel-covered region has no measured zero "
        "point."),

    ("extinction_adopted_source", "A_COL_K"): ("mag A_K",
        "For every cataloged source of this region, the extinction a "
        "star's own light passes through: the adopted dust column (this "
        "package's column_adopted_source A_COL_K, re-derived here) "
        "multiplied by this source's own F_EXTINCTION, which raises it to "
        "match the Juvela & Montillaud (2016) star-color map wherever "
        "that map reads higher."),
    ("extinction_adopted_source", "A_COL_SIG_K"): ("mag A_K",
        "The adopted dust column's own 1-sigma uncertainty, multiplied "
        "by this source's own F_EXTINCTION factor."),
    ("extinction_adopted_source", "A_COL_PROVENANCE"): ("code",
        "Which arm the dust column A_COL_K is scaled from, before the "
        "F_EXTINCTION correction: 0 Herschel, 1 Planck."),
    ("extinction_adopted_source", "A_COL_FWHM_ARCSEC"): ("arcsec",
        "The beam size of whichever map the dust column was taken from: "
        "the Herschel Gould Belt Survey's 36.3 arcsec where "
        "A_COL_PROVENANCE is 0, the measured Planck beam otherwise."),
    ("extinction_adopted_source", "HERSCHEL_MAP_ID"): ("map index",
        "For every cataloged source of this region, which map in this "
        "file's own MAP_NAME list the dust column was read from, as a "
        "position in that list. -1 where A_COL_PROVENANCE is 1 (Planck)."),
    ("extinction_adopted_source", "MAP_NAME"): ("map name",
        "The file name of each HGBS column-density map that reaches some "
        "part of this region, in the order HERSCHEL_MAP_ID indexes."),
    ("extinction_adopted_source", "ZP_SIGMA_K"): ("mag A_K",
        "For every cataloged source of this region, the uncertainty on "
        "the Herschel field zero-point offset already subtracted from the "
        "dust column before scaling: 0 where A_COL_PROVENANCE is 1 "
        "(Planck), or where the source's own Herschel-covered region has "
        "no measured zero point."),
    ("extinction_adopted_source", "F_EXTINCTION"): ("dimensionless",
        "For every cataloged source of this region, the factor its "
        "adopted dust column is multiplied by to form A_COL_K: at least "
        "1, the ratio of the Juvela & Montillaud (2016) star-color "
        "extinction to the adopted dust column, averaged over the "
        "source's own nside-1024 cell. 1 where that ratio is below 1, "
        "since a star-color map can read low but the emission map's own "
        "dust cannot be reduced by it."),

    ("column_adopted_sightline", "HPX_PIX_256"): ("nested HEALPix pixel, nside 256",
        "Every region's admitted nside-256 pixel numbers (nested "
        "ordering), concatenated region by region. Row i here is row i of "
        "A_K."),
    ("column_adopted_sightline", "REGION_CODE"): ("region code",
        "The region each pixel in HPX_PIX_256 belongs to, as the integer "
        "code the granule map assigns that region (its REGION_CODE_AXIS, "
        "where that dataset is present; otherwise its own region_code "
        "table)."),
    ("column_adopted_sightline", "A_K"): ("mag A_K",
        "For the matching pixel in HPX_PIX_256, the adopted dust-column "
        "extinction: the Herschel arm's block-averaged A_K where it "
        "covers that pixel, the Planck sightline column's A_K otherwise."),
    ("column_adopted_sightline", "SIGMA_A_K"): ("mag A_K",
        "The 1-sigma uncertainty on A_K, from whichever arm was adopted "
        "for that pixel: the Herschel sigma model (zero point, within-"
        "beam and random terms) where Herschel was used, the Planck "
        "sightline column's own uncertainty otherwise."),
    ("column_adopted_sightline", "PROVENANCE"): ("code",
        "Which arm A_K for the matching pixel came from: 0 Herschel, 1 "
        "Planck."),

    ("extinction_adopted_sightline", "HPX_PIX_256"): ("nested HEALPix pixel, nside 256",
        "Every region's admitted nside-256 pixel numbers (nested "
        "ordering), concatenated region by region. Row i here is row i of "
        "A_K."),
    ("extinction_adopted_sightline", "REGION_CODE"): ("region code",
        "The region each pixel in HPX_PIX_256 belongs to, as the integer "
        "code the granule map assigns that region."),
    ("extinction_adopted_sightline", "A_K"): ("mag A_K",
        "For the matching pixel in HPX_PIX_256, the extinction a star's "
        "own light passes through: this package's adopted sightline "
        "column (column_adopted_sightline A_K) multiplied by F_EXTINCTION."),
    ("extinction_adopted_sightline", "SIGMA_A_K"): ("mag A_K",
        "The adopted sightline column's own 1-sigma uncertainty, "
        "multiplied by this pixel's own F_EXTINCTION factor."),
    ("extinction_adopted_sightline", "F_EXTINCTION"): ("dimensionless",
        "For the matching pixel in HPX_PIX_256, the mean of its own "
        "cataloged sources' per-source F_EXTINCTION factor (this "
        "package's extinction_adopted_source F_EXTINCTION). 1 where the "
        "pixel is admitted but carries no cataloged source, so the "
        "adopted column passes through unscaled."),
    ("extinction_adopted_sightline", "PROVENANCE"): ("code",
        "Which arm the dust column before the F_EXTINCTION scaling came "
        "from, carried unchanged from the adopted sightline column: 0 "
        "Herschel, 1 Planck."),

    ("column-check_adopted_survey", "HPX_PIX_256"): ("nested HEALPix pixel, nside 256",
        "The Herschel-covered admitted sightline pixel numbers (nested "
        "ordering) this disagreement check is measured on, concatenated "
        "region by region. Row i here is row i of A_HERSCHEL."),
    ("column-check_adopted_survey", "REGION_CODE"): ("region code",
        "The region each pixel in HPX_PIX_256 belongs to, as the integer "
        "code the granule map assigns that region."),
    ("column-check_adopted_survey", "A_HERSCHEL"): ("mag A_K",
        "For the matching pixel in HPX_PIX_256, the Herschel arm's own "
        "block-averaged K-band extinction."),
    ("column-check_adopted_survey", "A_PLANCK"): ("mag A_K",
        "For the matching pixel in HPX_PIX_256, the Planck arm's own "
        "K-band extinction."),
    ("column-check_adopted_survey", "A_MAP_EDGE"): ("mag A_K",
        "For the matching pixel in HPX_PIX_256, the Edenhofer et al. "
        "(2023) 3-D extinction map's own cumulative extinction at its "
        "tabulated edge, before any far-field rescaling."),
    ("column-check_adopted_survey", "A_HERSCHEL_QUARTILE_EDGES"): ("mag A_K",
        "The five edges of the four quartile bins A_HERSCHEL is split "
        "into for BIN_MAP_EDGE_RATIO_* and BIN_PLANCK_RATIO_*: bin g runs "
        "from A_HERSCHEL_QUARTILE_EDGES[g] to [g+1]."),
    ("column-check_adopted_survey", "BIN_MAP_EDGE_RATIO_MEDIAN"): ("dimensionless",
        "For each of the four A_HERSCHEL quartile bins, the median, over "
        "that bin's own pixels, of A_MAP_EDGE / A_HERSCHEL."),
    ("column-check_adopted_survey", "BIN_MAP_EDGE_RATIO_P16"): ("dimensionless",
        "For each of the four A_HERSCHEL quartile bins, the 16th "
        "percentile, over that bin's own pixels, of A_MAP_EDGE / "
        "A_HERSCHEL."),
    ("column-check_adopted_survey", "BIN_MAP_EDGE_RATIO_P84"): ("dimensionless",
        "For each of the four A_HERSCHEL quartile bins, the 84th "
        "percentile, over that bin's own pixels, of A_MAP_EDGE / "
        "A_HERSCHEL."),
    ("column-check_adopted_survey", "BIN_PLANCK_RATIO_MEDIAN"): ("dimensionless",
        "For each of the four A_HERSCHEL quartile bins, the median, over "
        "that bin's own pixels, of A_PLANCK / A_HERSCHEL."),
    ("column-check_adopted_survey", "BIN_PLANCK_RATIO_P16"): ("dimensionless",
        "For each of the four A_HERSCHEL quartile bins, the 16th "
        "percentile, over that bin's own pixels, of A_PLANCK / "
        "A_HERSCHEL."),
    ("column-check_adopted_survey", "BIN_PLANCK_RATIO_P84"): ("dimensionless",
        "For each of the four A_HERSCHEL quartile bins, the 84th "
        "percentile, over that bin's own pixels, of A_PLANCK / "
        "A_HERSCHEL."),
    ("column-check_adopted_survey", "REGION"): ("region name",
        "Which region each row of REGION_MAP_EDGE_RATIO_* and "
        "REGION_PLANCK_RATIO_* describes, in the order REGION_CODE_AXIS "
        "gives."),
    ("column-check_adopted_survey", "REGION_CODE_AXIS"): ("region code",
        "The integer region code matching each entry of REGION."),
    ("column-check_adopted_survey", "REGION_MAP_EDGE_RATIO_MEDIAN"): ("dimensionless",
        "For the matching region in REGION, the median, over that "
        "region's own Herschel-covered admitted pixels, of A_MAP_EDGE / "
        "A_HERSCHEL."),
    ("column-check_adopted_survey", "REGION_MAP_EDGE_RATIO_P16"): ("dimensionless",
        "For the matching region in REGION, the 16th percentile, over "
        "that region's own Herschel-covered admitted pixels, of "
        "A_MAP_EDGE / A_HERSCHEL."),
    ("column-check_adopted_survey", "REGION_MAP_EDGE_RATIO_P84"): ("dimensionless",
        "For the matching region in REGION, the 84th percentile, over "
        "that region's own Herschel-covered admitted pixels, of "
        "A_MAP_EDGE / A_HERSCHEL."),
    ("column-check_adopted_survey", "REGION_PLANCK_RATIO_MEDIAN"): ("dimensionless",
        "For the matching region in REGION, the median, over that "
        "region's own Herschel-covered admitted pixels, of A_PLANCK / "
        "A_HERSCHEL."),
    ("column-check_adopted_survey", "REGION_PLANCK_RATIO_P16"): ("dimensionless",
        "For the matching region in REGION, the 16th percentile, over "
        "that region's own Herschel-covered admitted pixels, of "
        "A_PLANCK / A_HERSCHEL."),
    ("column-check_adopted_survey", "REGION_PLANCK_RATIO_P84"): ("dimensionless",
        "For the matching region in REGION, the 84th percentile, over "
        "that region's own Herschel-covered admitted pixels, of "
        "A_PLANCK / A_HERSCHEL."),

    # sky.derived.subbeam -- subbeam_herschel_region: how much the true
    # column varies within one Herschel beam, per region, plus the
    # column-conditional kernel and noise-separated structural mixture
    # the column prior's evaluator reads at an arbitrary beam.
    ("subbeam_herschel_region", "REGION"): ("region name",
        "Which Herschel-covered region each row of every other dataset in "
        "this file describes. Only regions with HGBS coverage appear; "
        "this is not the fixed thirty-region table."),
    ("subbeam_herschel_region", "SCALES"): ("arcsec",
        "The beam sizes of the two-scale increment ladder every region's "
        "COND_QUANTILES row is measured at, shared by every region in "
        "this file."),
    ("subbeam_herschel_region", "QS"): ("dimensionless",
        "The seven probability levels (0.01, 0.05, 0.16, 0.50, 0.84, 0.95, "
        "0.99) COND_QUANTILES is tabulated at."),
    ("subbeam_herschel_region", "BETA"): ("dimensionless",
        "The fitted power-law index beta of this region's true-column "
        "power spectrum P(k) ~ k^-beta, from its own beam-ladder two-scale "
        "increments."),
    ("subbeam_herschel_region", "BETA_BOOT_P16"): ("dimensionless",
        "The 16th percentile of BETA over 200 block-bootstrap resamples of "
        "this region's own 512-arcsec tiles: the fit's own lower sampling "
        "bound."),
    ("subbeam_herschel_region", "BETA_BOOT_P84"): ("dimensionless",
        "The 84th percentile of BETA over 200 block-bootstrap resamples "
        "of this region's own 512-arcsec tiles: the fit's own upper "
        "sampling bound."),
    ("subbeam_herschel_region", "W_ABS_36P3"): ("dimensionless (ln column ratio)",
        "The median, over 200 block-bootstrap resamples of this "
        "region's own 512-arcsec tiles, of the absolute pencil-to-beam "
        "width of true column at the HGBS map's own 36.3 arcsec beam, "
        "in natural-log column units."),
    ("subbeam_herschel_region", "W_ABS_L108"): ("dimensionless (ln column ratio)",
        "The median, over 200 block-bootstrap resamples of this region's "
        "own 512-arcsec tiles, of the absolute pencil-to-beam width of "
        "true column at a 108 arcsec beam, in natural-log column units."),
    ("subbeam_herschel_region", "W_ABS_L302"): ("dimensionless (ln column ratio)",
        "The median, over 200 block-bootstrap resamples of this region's "
        "own 512-arcsec tiles, of the absolute pencil-to-beam width of "
        "true column at a 301.8 arcsec beam (Planck's own measured "
        "beam), in natural-log column units."),
    ("subbeam_herschel_region", "W_ABS_L821"): ("dimensionless (ln column ratio)",
        "The median, over 200 block-bootstrap resamples of this region's "
        "own 512-arcsec tiles, of the absolute pencil-to-beam width of "
        "true column at an 821 arcsec beam (the extinction-profile "
        "grid's nside-256 pixel scale), in natural-log column units."),
    ("subbeam_herschel_region", "COMPLETION_L108"): ("dimensionless",
        "W_ABS_L108 divided by the two-scale width actually measured "
        "between the map's own beam and 108 arcsec: how much of the "
        "absolute pencil-to-beam width at 108 arcsec is directly measured "
        "rather than model extrapolation."),
    ("subbeam_herschel_region", "COMPLETION_L302"): ("dimensionless",
        "W_ABS_L302 divided by the two-scale width actually measured "
        "between the map's own beam and 301.8 arcsec: how much of the "
        "absolute pencil-to-beam width at 301.8 arcsec is directly "
        "measured rather than model extrapolation."),
    ("subbeam_herschel_region", "COMPLETION_L821"): ("dimensionless",
        "W_ABS_L821 divided by the two-scale width actually measured "
        "between the map's own beam and 821 arcsec: how much of the "
        "absolute pencil-to-beam width at 821 arcsec is directly "
        "measured rather than model extrapolation."),
    ("subbeam_herschel_region", "RESCALE_EXPONENT"): ("dimensionless",
        "(BETA - 2) / 2: the exponent that rescales the absolute "
        "pencil-to-beam width from one beam size to another under this "
        "region's fitted power spectrum."),
    ("subbeam_herschel_region", "OFFSET_EXPONENT"): ("dimensionless",
        "The power-law exponent p of the two-scale distribution's median "
        "offset from zero, offset(L) = -c*L^p, fit through the measured "
        "median offsets at 301.8 and 821 arcsec (c itself is not stored)."),
    ("subbeam_herschel_region", "N_FIT"): ("pairs",
        "How many measured beam pairs at or above 108 arcsec went into "
        "this region's held-out-scale validation fit."),
    ("subbeam_herschel_region", "N_PRED"): ("pairs",
        "How many finer, held-out beam pairs the validation fit's own "
        "prediction (RMS_PRED_DEX) was checked against."),
    ("subbeam_herschel_region", "RMS_PRED_DEX"): ("dex",
        "The root-mean-square error, in dex, of the held-out-scale "
        "validation: the fit from N_FIT's own coarse pairs predicting "
        "N_PRED's own finer pairs."),
    ("subbeam_herschel_region", "COND_QUANTILES"): ("dimensionless (ln column ratio)",
        "For every region, every beam size in SCALES and every "
        "probability level in QS, the quantile of d = ln(A at the map's "
        "native beam) - ln(A at that ladder beam), the two-scale "
        "increment this region's power-spectrum fit is measured from."),
    ("subbeam_herschel_region", "KA_EDGES"): ("ln(mag A_K)",
        "The edges of the natural-log column bins the COND_KERNEL_* "
        "histograms' first axis runs over."),
    ("subbeam_herschel_region", "KD_EDGES"): ("dimensionless (ln column ratio)",
        "The edges of the two-scale increment d's own bins, the "
        "COND_KERNEL_* histograms' second axis."),
    ("subbeam_herschel_region", "COND_KERNEL_L108"): ("counts",
        "For every region, the 2-D histogram of ln(true column) (KA_EDGES) "
        "against the two-scale increment d to a 108 arcsec beam "
        "(KD_EDGES), summed over that region's own HGBS maps: the column-"
        "conditional kernel at 108 arcsec."),
    ("subbeam_herschel_region", "COND_KERNEL_L302"): ("counts",
        "For every region, the 2-D histogram of ln(true column) "
        "(KA_EDGES) against the two-scale increment d to a 301.8 "
        "arcsec beam (Planck's own measured beam) (KD_EDGES), summed "
        "over that region's own HGBS maps: the column-conditional "
        "kernel at 301.8 arcsec."),
    ("subbeam_herschel_region", "COND_KERNEL_L821"): ("counts",
        "For every region, the 2-D histogram of ln(true column) "
        "(KA_EDGES) against the two-scale increment d to an 821 arcsec "
        "beam (the extinction-profile grid's nside-256 pixel scale) "
        "(KD_EDGES), summed over that region's own HGBS maps: the "
        "column-conditional kernel at 821 arcsec."),
    ("subbeam_herschel_region", "MIX_W"): ("dimensionless",
        "For every region, beam (108, 301.8, 821 arcsec) and column bin of "
        "MIX_KA_CENTRES with at least 200 counts, the mixing weight of the "
        "first of two Gaussian components in the noise-separated "
        "structural mixture fit to that bin's own column-conditional "
        "kernel, after deconvolving the fine map's own noise "
        "(FINE_MAP_NOISE_K). NaN where that bin had too few counts."),
    ("subbeam_herschel_region", "MIX_MU1"): ("dimensionless (ln column ratio)",
        "For every region, beam and column bin of MIX_KA_CENTRES with "
        "at least 200 counts, the mean of the first of the two Gaussian "
        "components in the noise-separated structural mixture fit to "
        "that bin's own column-conditional kernel. NaN where that bin "
        "had too few counts."),
    ("subbeam_herschel_region", "MIX_MU2"): ("dimensionless (ln column ratio)",
        "For every region, beam and column bin of MIX_KA_CENTRES with "
        "at least 200 counts, the mean of the second of the two "
        "Gaussian components in the noise-separated structural mixture "
        "fit to that bin's own column-conditional kernel. NaN where "
        "that bin had too few counts."),
    ("subbeam_herschel_region", "MIX_SIG1"): ("dimensionless (ln column ratio)",
        "For every region, beam and column bin of MIX_KA_CENTRES with "
        "at least 200 counts, the standard deviation of the first of "
        "the two Gaussian components in the noise-separated structural "
        "mixture fit to that bin's own column-conditional kernel. NaN "
        "where that bin had too few counts."),
    ("subbeam_herschel_region", "MIX_SIG2"): ("dimensionless (ln column ratio)",
        "For every region, beam and column bin of MIX_KA_CENTRES with "
        "at least 200 counts, the standard deviation of the second of "
        "the two Gaussian components in the noise-separated structural "
        "mixture fit to that bin's own column-conditional kernel. NaN "
        "where that bin had too few counts."),
    ("subbeam_herschel_region", "MIX_MAX_CDF_ERR"): ("dimensionless",
        "For every region, beam and column bin of MIX_KA_CENTRES with "
        "at least 200 counts, the largest absolute difference, over "
        "the fitted histogram's own bins, between the noise-convolved "
        "mixture model's CDF and the observed CDF: that fit's own "
        "worst-case goodness-of-fit error. NaN where that bin had too "
        "few counts."),
    ("subbeam_herschel_region", "MIX_KA_CENTRES"): ("ln(mag A_K)",
        "The center of each natural-log column bin the mixture-fit "
        "datasets in this file are tabulated on: the midpoints of "
        "KA_EDGES."),
    ("subbeam_herschel_region", "FINE_MAP_NOISE_K"): ("mag A_K",
        "For every region, the estimated standard deviation of the "
        "36.3 arcsec HGBS map's own additive noise, in A_K units, from the "
        "lowest well-populated column bin of its 108 arcsec conditional "
        "histogram."),
    ("subbeam_herschel_region", "MIX_POOLED_W"): ("dimensionless",
        "For every beam (108, 301.8, 821 arcsec) and column bin of "
        "MIX_KA_CENTRES, the mixing weight of the first of two Gaussian "
        "components in the noise-separated structural mixture fit to "
        "that bin's own column-conditional kernel summed over every "
        "region's counts, after deconvolving each region's own fine-"
        "map noise."),
    ("subbeam_herschel_region", "MIX_POOLED_MU1"): ("dimensionless (ln column ratio)",
        "For every beam (108, 301.8, 821 arcsec) and column bin of "
        "MIX_KA_CENTRES, the mean of the first of two Gaussian "
        "components in the noise-separated structural mixture fit to "
        "that bin's own column-conditional kernel summed over every "
        "region's counts, after deconvolving each region's own fine-"
        "map noise."),
    ("subbeam_herschel_region", "MIX_POOLED_MU2"): ("dimensionless (ln column ratio)",
        "For every beam (108, 301.8, 821 arcsec) and column bin of "
        "MIX_KA_CENTRES, the mean of the second of two Gaussian "
        "components in the noise-separated structural mixture fit to "
        "that bin's own column-conditional kernel summed over every "
        "region's counts, after deconvolving each region's own fine-"
        "map noise."),
    ("subbeam_herschel_region", "MIX_POOLED_SIG1"): ("dimensionless (ln column ratio)",
        "For every beam (108, 301.8, 821 arcsec) and column bin of "
        "MIX_KA_CENTRES, the standard deviation of the first of two "
        "Gaussian components in the noise-separated structural mixture "
        "fit to that bin's own column-conditional kernel summed over "
        "every region's counts, after deconvolving each region's own "
        "fine-map noise."),
    ("subbeam_herschel_region", "MIX_POOLED_SIG2"): ("dimensionless (ln column ratio)",
        "For every beam (108, 301.8, 821 arcsec) and column bin of "
        "MIX_KA_CENTRES, the standard deviation of the second of two "
        "Gaussian components in the noise-separated structural mixture "
        "fit to that bin's own column-conditional kernel summed over "
        "every region's counts, after deconvolving each region's own "
        "fine-map noise."),
    ("subbeam_herschel_region", "MIX_POOLED_MAX_CDF_ERR"): ("dimensionless",
        "For every beam (108, 301.8, 821 arcsec) and column bin of "
        "MIX_KA_CENTRES, the largest absolute difference, over the "
        "fitted histogram's own bins, between the noise-convolved "
        "mixture model's CDF and the observed CDF, for the structural "
        "mixture fit to that bin's own column-conditional kernel "
        "summed over every region's counts: that fit's own worst-case "
        "goodness-of-fit error."),

    # sky.derived.planck_column -- calibration_planck_survey (survey-wide
    # tau353-to-A_K calibration) and column_planck_sightline (per nside-256
    # pixel): the Planck arm of the extinction column.
    ("calibration_planck_survey", "REGION"): ("region name",
        "Which HGBS-covered region each row of CV_FRAC_ERROR describes."),
    ("calibration_planck_survey", "CV_FRAC_ERROR"): ("dimensionless",
        "For the matching region in REGION, the median fractional "
        "difference between its own HGBS column and the Planck-based "
        "column predicted for it by the calibration fit with that region "
        "held out: the cross-validation residual this file's own "
        "SIGMA_REGION_FRAC attribute is the scatter of."),

    ("column_planck_sightline", "HPX_PIX_256"): ("nested HEALPix pixel, nside 256",
        "This file's admitted nside-256 pixel numbers (nested ordering, "
        "the granule map's own footprint). Row i here is row i of A_K."),
    ("column_planck_sightline", "A_K"): ("mag A_K",
        "For the matching pixel in HPX_PIX_256, the K-band extinction from "
        "the Planck thermal-dust optical depth (TAU353) at that pixel, "
        "through this file's own A_TAU calibration."),
    ("column_planck_sightline", "SIGMA_A_K"): ("mag A_K",
        "The total 1-sigma uncertainty on A_K for the matching pixel, "
        "combining the calibration's own statistical, within-beam and "
        "region-to-region terms in quadrature."),
    ("column_planck_sightline", "GAL_L_DEG"): ("deg",
        "Galactic longitude of the matching pixel's center."),
    ("column_planck_sightline", "GAL_B_DEG"): ("deg",
        "Galactic latitude of the matching pixel's center."),
    ("column_planck_sightline", "TEMP_K"): ("K",
        "Planck's own fitted thermal-dust temperature at the matching "
        "pixel."),

    # sky.derived.dunham_yso -- yso_dunham2015_survey: the Dunham et al.
    # (2015) YSO census, one row per cataloged YSO.
    ("yso_dunham2015_survey", "SEQ"): ("catalog sequence number",
        "Dunham et al. (2015)'s own running sequence number for this YSO, "
        "the order every dataset in this file shares."),
    ("yso_dunham2015_survey", "CLOUD"): ("cloud name",
        "Which of the eighteen clouds of Dunham et al. (2015) this YSO "
        "belongs to."),
    ("yso_dunham2015_survey", "ID"): ("catalog identifier",
        "This YSO's own Spitzer source name, the J2000 sexagesimal string "
        "RA_DEG and DEC_DEG are parsed from."),
    ("yso_dunham2015_survey", "RA_DEG"): ("deg",
        "Right ascension, equinox J2000, parsed from this row's own ID."),
    ("yso_dunham2015_survey", "DEC_DEG"): ("deg",
        "Declination, equinox J2000, parsed from this row's own ID."),
    ("yso_dunham2015_survey", "DIST_PC"): ("pc",
        "The adopted distance to this YSO's own cloud, Dunham et al. "
        "(2015)'s Table 1."),
    ("yso_dunham2015_survey", "AV"): ("mag A_V",
        "This YSO's fitted visual extinction, Dunham et al. (2015)'s own "
        "SED fit."),
    ("yso_dunham2015_survey", "ALPHA0"): ("dimensionless",
        "This YSO's infrared spectral index (d log(lambda F_lambda) / "
        "d log(lambda)) before extinction correction, Dunham et al. "
        "(2015)'s own measurement."),
    ("yso_dunham2015_survey", "TBOL0_K"): ("K",
        "This YSO's bolometric temperature before extinction correction, "
        "Dunham et al. (2015)'s own measurement."),
    ("yso_dunham2015_survey", "LBOL0_LSUN"): ("Lsun",
        "This YSO's bolometric luminosity before extinction correction, "
        "Dunham et al. (2015)'s own measurement."),
    ("yso_dunham2015_survey", "F45_MJY"): ("mJy",
        "This YSO's extinction-corrected 4.5 micron flux density, Dunham "
        "et al. (2015)'s Table 4."),
    ("yso_dunham2015_survey", "E_F45_MJY"): ("mJy",
        "The 1-sigma uncertainty on F45_MJY, Dunham et al. (2015)'s "
        "Table 4."),
    ("yso_dunham2015_survey", "F45_OBS_MJY"): ("mJy",
        "This YSO's observed (not extinction-corrected) 4.5 micron flux "
        "density, Dunham et al. (2015)'s Table 3."),
    ("yso_dunham2015_survey", "F36_MJY"): ("mJy",
        "This YSO's extinction-corrected 3.6 micron flux density, Dunham "
        "et al. (2015)'s Table 4."),
    ("yso_dunham2015_survey", "F58_MJY"): ("mJy",
        "This YSO's extinction-corrected 5.8 micron flux density, Dunham "
        "et al. (2015)'s Table 4."),
    ("yso_dunham2015_survey", "F80_MJY"): ("mJy",
        "This YSO's extinction-corrected 8.0 micron flux density, Dunham "
        "et al. (2015)'s Table 4."),
    ("yso_dunham2015_survey", "F24_MJY"): ("mJy",
        "This YSO's extinction-corrected 24 micron flux density, Dunham "
        "et al. (2015)'s Table 4."),
    ("yso_dunham2015_survey", "LOG10_F45_REF"): ("log10 mJy at 1 kpc",
        "log10 of F45_MJY scaled to a common reference distance of 1 kpc "
        "by (DIST_PC / 1000)^2: 10**x is a flux in mJy a source at this "
        "YSO's own brightness would show at 1 kpc. This is the "
        "reference-brightness scale and distance the YSO templates "
        "used elsewhere in this package are also tabulated at."),

    # sky.derived.herschel_column -- sigma_herschel_survey (survey-wide
    # calibration) and column_herschel_source (per region): the Herschel
    # Gould Belt Survey dust-column arm of the per-source extinction.
    ("sigma_herschel_survey", "MAP_NAME"): ("map name",
        "The file name of each HGBS column-density map this survey-wide "
        "calibration was measured from."),
    ("sigma_herschel_survey", "BEAM_FWHM_ARCSEC"): ("arcsec",
        "The matching map's own beam full width at half maximum, from its "
        "FITS header where stated, else the survey's own stated value, "
        "36.3 arcsec."),
    ("sigma_herschel_survey", "SIGMA_ZP_K"): ("mag A_K",
        "The root-mean-square, over every field with a measured zero "
        "point in ZP_FIELD, of that field's own zero point: one number "
        "for a reader that wants a single survey-wide zero-point "
        "uncertainty rather than FIELD_NAME's per-field values."),
    ("sigma_herschel_survey", "C0"): ("mag A_K",
        "The constant term of the random-uncertainty model SIGMA_RAND_K^2 "
        "= C0^2 + C1^2 * A_K^2, fit over every overlapping HGBS map pair's "
        "own column-binned scatter."),
    ("sigma_herschel_survey", "C1"): ("dimensionless",
        "The term of the random-uncertainty model SIGMA_RAND_K^2 = C0^2 + "
        "C1^2 * A_K^2 that scales with A_K, fit over every overlapping "
        "HGBS map pair's own column-binned scatter."),
    ("sigma_herschel_survey", "N_PAIRS"): ("map pairs",
        "How many overlapping HGBS map pairs went into fitting C0 and C1."),
    ("sigma_herschel_survey", "FIELD_NAME"): ("region name",
        "Which SESNA region (HGBS-covered \"field\") each row of ZP_FIELD, "
        "ZP_SIGMA_FIELD and ZP_N_FIELD describes."),
    ("sigma_herschel_survey", "ZP_FIELD"): ("mag A_K",
        "For the matching field in FIELD_NAME, the median of the Herschel "
        "column minus the Planck column over that field's own low-column "
        "(A_PLANCK < 0.3 mag), Herschel-covered admitted sightline pixels: "
        "this field's own Herschel zero-point offset."),
    ("sigma_herschel_survey", "ZP_SIGMA_FIELD"): ("mag A_K",
        "For the matching field in FIELD_NAME, the uncertainty on "
        "ZP_FIELD: the robust scatter (1.4826 times the median absolute "
        "deviation) of the Herschel column minus the Planck column over "
        "that field's own low-column, Herschel-covered admitted "
        "sightline pixels, divided by the square root of ZP_N_FIELD."),
    ("sigma_herschel_survey", "ZP_N_FIELD"): ("pixels",
        "For the matching field in FIELD_NAME, how many low-column, "
        "Herschel-covered admitted sightline pixels went into ZP_FIELD and "
        "ZP_SIGMA_FIELD."),

    ("column_herschel_source", "A_K"): ("mag A_K",
        "For every cataloged source of this region, the K-band "
        "extinction from the Herschel Gould Belt Survey N(H2) map at the "
        "source's own position, where a map covers it. Not meaningful "
        "where COVERED is False."),
    ("column_herschel_source", "SIGMA_A_K"): ("mag A_K",
        "The total 1-sigma uncertainty on A_K, combining SIGMA_RAND_K and "
        "this map's own zero-point uncertainty in quadrature. Not "
        "meaningful where COVERED is False."),
    ("column_herschel_source", "SIGMA_RAND_K"): ("mag A_K",
        "The part of A_K's uncertainty from map noise that varies "
        "independently from source to source: sqrt(C0^2 + C1^2 * A_K^2), "
        "the survey-wide calibration's own random-uncertainty model. Not "
        "meaningful where COVERED is False."),
    ("column_herschel_source", "SIGMA_ZP_K"): ("mag A_K",
        "The part of A_K's uncertainty from this source's own map's "
        "zero-point offset, a systematic shared by every source that map "
        "covers. Not meaningful where COVERED is False."),
    ("column_herschel_source", "COVERED"): ("boolean",
        "True where an HGBS map reaches this source's own position, so "
        "A_K and its uncertainty columns in this file are meaningful."),
    ("column_herschel_source", "MAP_ID"): ("map index",
        "For every cataloged source of this region, which map in this "
        "file's own MAP_NAME list covers its position, as a position in "
        "that list. -1 where no map covers it."),
    ("column_herschel_source", "MAP_NAME"): ("map name",
        "The file name of each HGBS column-density map that reaches some "
        "part of this region, in the order MAP_ID indexes."),

    # sky.derived.protostars -- protostars_survey: the pooled HOPS/eHOPS
    # Herschel-confirmed protostar sample, a report-only overlay.
    ("protostars_survey", "SURVEY"): ("survey name",
        "Which survey this protostar comes from: HOPS (Furlan et al. 2016, "
        "Orion) or eHOPS (Pokhrel et al. 2023, Aquila)."),
    ("protostars_survey", "ID"): ("catalog identifier",
        "This protostar's own identifier in its source survey's table."),
    ("protostars_survey", "RA_DEG"): ("deg", "Right ascension, equinox J2000."),
    ("protostars_survey", "DEC_DEG"): ("deg", "Declination, equinox J2000."),
    ("protostars_survey", "CLASS"): ("code",
        "This protostar's evolutionary class in its source survey's own "
        "SED-fitted classification, on HOPS's own four-way scale: \"0\" "
        "Class 0, \"I\" Class I, \"flat\" flat-spectrum, \"II\" Class II. "
        "eHOPS's own labels are mapped onto these four before being "
        "written here."),
    ("protostars_survey", "LBOL_LSUN"): ("Lsun",
        "This protostar's bolometric luminosity, from its source survey's "
        "own SED fit."),
    ("protostars_survey", "TBOL_K"): ("K",
        "This protostar's bolometric temperature, from its source survey's "
        "own SED fit."),
    ("protostars_survey", "AV_FOREGROUND_MAG"): ("mag A_V",
        "This protostar's foreground V-band extinction, from its source "
        "survey's own SED fit."),
    ("protostars_survey", "F45_MJY"): ("mJy",
        "This protostar's measured 4.5 micron flux density, from its "
        "source survey's own photometry table. NaN where F45_MEASURED is "
        "False."),
    ("protostars_survey", "E_F45_MJY"): ("mJy",
        "The 1-sigma uncertainty on F45_MJY. NaN where F45_MEASURED is "
        "False."),
    ("protostars_survey", "F45_MEASURED"): ("boolean",
        "True where this protostar's source survey reports an actual 4.5 "
        "micron flux measurement rather than no photometry at that "
        "wavelength."),
    ("protostars_survey", "REGION"): ("region name",
        "The SESNA region whose admitted footprint contains this "
        "protostar's position, or an empty string where its position "
        "falls outside every region's footprint."),

    # sky.derived.gaia_match -- match_gaia_source: the SESNA-Gaia crossmatch
    # and its congruence term G_S, per source.
    ("match_gaia_source", "G_S"): ("dimensionless",
        "The Gaia congruence term for this source: the likelihood ratio of "
        "its Gaia match data under a true counterpart versus chance "
        "alignment, renormalized at even prior odds (L / (1+L)). Close to "
        "1 means the match data strongly favor a true Gaia counterpart; "
        "close to 0.5 means the two are equally favored; low values favor "
        "chance alignment, including the case of no candidate at all."),
    ("match_gaia_source", "SEP_ARCSEC"): ("arcsec",
        "The angular separation to the nearest Gaia DR3 candidate within 3 "
        "arcsec of this source, after propagating that candidate's own "
        "proper motion (where it has one) to the survey epoch. NaN where "
        "no Gaia candidate falls within that radius."),
    ("match_gaia_source", "GAIA_SOURCE_ID"): ("Gaia DR3 source_id",
        "The matched candidate's own Gaia DR3 catalog identifier. -1 "
        "where no candidate falls within SEP_ARCSEC's 3 arcsec radius."),
    ("match_gaia_source", "G_MAG"): ("mag Gaia G",
        "The matched candidate's Gaia G-band magnitude. NaN where no "
        "candidate falls within SEP_ARCSEC's 3 arcsec radius."),
    ("match_gaia_source", "BP_MAG"): ("mag Gaia BP",
        "The matched candidate's Gaia BP-band magnitude. NaN where no "
        "candidate falls within SEP_ARCSEC's 3 arcsec radius."),
    ("match_gaia_source", "RP_MAG"): ("mag Gaia RP",
        "The matched candidate's Gaia RP-band magnitude. NaN where no "
        "candidate falls within SEP_ARCSEC's 3 arcsec radius."),
    ("match_gaia_source", "PLX_MAS"): ("mas",
        "The matched candidate's Gaia parallax. NaN where no candidate "
        "falls within SEP_ARCSEC's 3 arcsec radius."),
    ("match_gaia_source", "E_PLX_MAS"): ("mas",
        "The 1-sigma uncertainty on PLX_MAS. NaN where no candidate falls "
        "within SEP_ARCSEC's 3 arcsec radius."),
    ("match_gaia_source", "RUWE"): ("dimensionless",
        "The matched candidate's Gaia renormalized unit weight error, a "
        "quality indicator for its astrometric solution (values well "
        "above 1.4 suggest a poor single-star fit). NaN where no "
        "candidate falls within SEP_ARCSEC's 3 arcsec radius."),
    ("match_gaia_source", "NO_PM"): ("boolean",
        "True where the matched candidate carries no Gaia proper-motion "
        "solution, so it was matched at the Gaia table's own 2016.0 epoch "
        "directly rather than propagated to the survey epoch. False where "
        "there is no match at all."),
    ("match_gaia_source", "RHO_PER_ARCSEC2"): ("per arcsec^2",
        "The local all-sky Gaia source surface density (G < 21) at this "
        "source's own nside-512 pixel, read from the Gaia anchor-count "
        "product: that pixel's Gaia star count divided by its solid "
        "angle, converted to square arcseconds. Used as the chance-"
        "alignment rate in G_S's own likelihood ratio, defined for every "
        "source regardless of whether it has a matched candidate."),

    # sky.derived.profile -- profile_edenhofer_sightline (per region) and
    # depth_edenhofer_region (survey-wide): the Edenhofer et al. (2023/2024)
    # cumulative extinction profile along each sightline, and each region's
    # own cloud depth read off it.
    ("profile_edenhofer_sightline", "HPX_PIX_256"): ("nested HEALPix pixel, nside 256",
        "This region's own admitted nside-256 pixel numbers (nested "
        "ordering). Row i here is row i of A_CUM_K and every other "
        "per-sightline dataset in this file."),
    ("profile_edenhofer_sightline", "DIST_PC"): ("pc",
        "The distance grid along the line of sight, starting at 0 pc, that "
        "every per-sightline dataset in this file is tabulated on."),
    ("profile_edenhofer_sightline", "A_CUM_K"): ("mag A_K",
        "For the matching pixel in HPX_PIX_256 and each distance in "
        "DIST_PC, the cumulative K-band extinction from the observer out "
        "to that distance along that sightline."),
    ("profile_edenhofer_sightline", "RHO_K_PER_PC"): ("mag A_K per pc",
        "For the matching pixel in HPX_PIX_256 and each interval between "
        "two adjacent points of DIST_PC, the mean rate of change of "
        "A_CUM_K over that interval: the local extinction density."),
    ("profile_edenhofer_sightline", "SIGMA_UNC_K"): ("mag A_K",
        "For the matching pixel in HPX_PIX_256 and each distance in "
        "DIST_PC, the lower-bound uncertainty on A_CUM_K: the sum, over "
        "every step out to that distance, of that step's own map "
        "uncertainty treated as independent of every other step's."),
    ("profile_edenhofer_sightline", "SIGMA_COR_K"): ("mag A_K",
        "For the matching pixel in HPX_PIX_256 and each distance in "
        "DIST_PC, the upper-bound uncertainty on A_CUM_K: the sum, over "
        "every step out to that distance, of that step's own map "
        "uncertainty treated as perfectly correlated with every other "
        "step's."),
    ("profile_edenhofer_sightline", "GAL_L_DEG"): ("deg",
        "Galactic longitude of the matching pixel's center in HPX_PIX_256."),
    ("profile_edenhofer_sightline", "GAL_B_DEG"): ("deg",
        "Galactic latitude of the matching pixel's center in HPX_PIX_256."),
    ("profile_edenhofer_sightline", "TAIL_RESIDUAL_K"): ("mag A_K",
        "For the matching pixel in HPX_PIX_256, the part of this "
        "sightline's own total column (A_INF_K) beyond the map's own "
        "tabulated edge that is attributed to the exponential dust-disc "
        "tail there, capped at what the map's own edge density and tail "
        "shape together imply; any remaining excess is spread back into "
        "A_CUM_K's own in-map values instead (see RESCALED)."),
    ("profile_edenhofer_sightline", "TAIL_MODE"): ("code",
        "For the matching pixel in HPX_PIX_256, which analytic shape the "
        "exponential dust-disc tail beyond the map's own edge takes along "
        "this sightline: 0 vertical (scale height above the Galactic "
        "plane), 1 disc (radial profile within the plane)."),
    ("profile_edenhofer_sightline", "TAIL_SCALE_PC"): ("pc",
        "For the matching pixel in HPX_PIX_256, the tail's own geometric "
        "scale along this sightline: the vertical scale height where "
        "TAIL_MODE is 0, or NaN where TAIL_MODE is 1 (the disc branch "
        "carries no single scale length)."),
    ("profile_edenhofer_sightline", "TAIL_EFOLD_PC"): ("pc",
        "For the matching pixel in HPX_PIX_256, the e-folding distance of "
        "the tail's exponential falloff along this sightline, consistent "
        "with TAIL_MODE and used to form TAIL_RESIDUAL_K."),
    ("profile_edenhofer_sightline", "A_INF_K"): ("mag A_K",
        "For the matching pixel in HPX_PIX_256, this sightline's own total "
        "K-band column from the adopted sightline column product "
        "(Herschel where covered, Planck elsewhere): the value A_CUM_K "
        "approaches as distance grows without bound."),
    ("profile_edenhofer_sightline", "RESCALED"): ("boolean",
        "For the matching pixel in HPX_PIX_256, True where this "
        "sightline's in-map profile (A_CUM_K and RHO_K_PER_PC) was scaled "
        "so its own edge value, plus TAIL_RESIDUAL_K, matches A_INF_K, "
        "rather than left as the input maps measured it."),
    ("profile_edenhofer_sightline", "fallback/DIST_PC"): ("pc",
        "The distance grid along the line of sight, starting at 0 pc, "
        "that fallback/A_CUM_K and the region's other fallback-profile "
        "datasets in this file are tabulated on."),
    ("profile_edenhofer_sightline", "fallback/A_CUM_K"): ("mag A_K",
        "For each distance in fallback/DIST_PC, this region's "
        "source-weighted mean of A_CUM_K over every sightline in this "
        "file, each sightline weighted by its own admitted-pixel source "
        "count: the one profile a consumer uses for a position with no "
        "sightline of its own in this file."),
    ("profile_edenhofer_sightline", "fallback/A_EDGE_K"): ("mag A_K",
        "This region's source-weighted mean, over every sightline in this "
        "file, of A_CUM_K's own value at the map's farthest tabulated "
        "distance."),
    ("profile_edenhofer_sightline", "fallback/A_COL_SIGHTLINE_K"): ("mag A_K",
        "This region's source-weighted mean, over every sightline in this "
        "file, of A_INF_K."),
    ("profile_edenhofer_sightline", "fallback/RESIDUAL_K"): ("mag A_K",
        "This region's source-weighted mean, over every sightline in this "
        "file, of TAIL_RESIDUAL_K."),
    ("profile_edenhofer_sightline", "fallback/GAL_L_DEG"): ("deg",
        "The Galactic longitude of this region's source-weighted mean sky "
        "direction, the direction the fallback tail geometry below is "
        "evaluated at."),
    ("profile_edenhofer_sightline", "fallback/GAL_B_DEG"): ("deg",
        "The Galactic latitude of this region's source-weighted mean sky "
        "direction, the direction the fallback tail geometry below is "
        "evaluated at."),
    ("profile_edenhofer_sightline", "fallback/TAIL_MODE"): ("code",
        "The tail shape (TAIL_MODE's own codes) evaluated at this region's "
        "source-weighted mean sky direction."),
    ("profile_edenhofer_sightline", "fallback/TAIL_SCALE_PC"): ("pc",
        "The tail scale length (TAIL_SCALE_PC) evaluated at this region's "
        "source-weighted mean sky direction."),
    ("profile_edenhofer_sightline", "fallback/TAIL_EFOLD_PC"): ("pc",
        "The tail e-folding distance (TAIL_EFOLD_PC) evaluated at this "
        "region's source-weighted mean sky direction."),

    ("depth_edenhofer_region", "REGION"): ("region name",
        "The name of the region this row describes, in the fixed thirty-"
        "region order every region-axis product in this package shares."),
    ("depth_edenhofer_region", "D_R_PC"): ("pc",
        "This region's own canonical distance, carried over from this "
        "package's region table."),
    ("depth_edenhofer_region", "SIGMA_D_PC"): ("pc",
        "The quoted uncertainty on this region's canonical distance, "
        "carried over from this package's region table."),
    ("depth_edenhofer_region", "D_PEAK_PC"): ("pc",
        "The distance of the extinction-weighted peak of this region's own "
        "fallback profile, in the structure nearest its canonical distance "
        "D_R_PC."),
    ("depth_edenhofer_region", "D_LO_PC"): ("pc",
        "The 16th-percentile distance of the extinction-weighted "
        "distribution inside that same structure."),
    ("depth_edenhofer_region", "D_HI_PC"): ("pc",
        "The 84th-percentile distance of the extinction-weighted "
        "distribution inside that same structure."),
    ("depth_edenhofer_region", "SIGMA_DEPTH_PC"): ("pc",
        "Half of D_HI_PC minus D_LO_PC: this region's own line-of-sight "
        "depth, read as a half-width."),
    ("depth_edenhofer_region", "FWHM_PC"): ("pc",
        "The full width at half maximum of the extinction-weighted "
        "distribution inside that same structure."),
    ("depth_edenhofer_region", "DEPTH_OK"): ("boolean",
        "True where a structure near this region's canonical distance was "
        "found at all, so D_PEAK_PC and the other depth values in this row "
        "are meaningful; False where none was found."),

    # sky.derived.twomass_column_scale -- column-scale_twomass_region: each
    # arm's map column regressed against 2MASS background-star color, by
    # region and arm.
    ("column-scale_twomass_region", "REGION"): ("region name",
        "The name of the region this row describes, in the fixed thirty-"
        "region order every region-axis product in this package shares."),
    ("column-scale_twomass_region", "ARM"): ("arm name",
        "The two map arms this file's second axis runs over, in column "
        "order: herschel, planck."),
    ("column-scale_twomass_region", "SCALE"): ("dimensionless",
        "For this region and the matching arm in ARM, the factor that "
        "rescales that arm's own map column to match the reddening of "
        "2MASS background stars: the regression slope of their median "
        "H minus Ks color against the arm's per-pixel column, divided by "
        "the adopted hybrid extinction law's own A_H/A_K minus 1. NaN "
        "where the region has too few usable pixels."),
    ("column-scale_twomass_region", "SCALE_SIGMA"): ("dimensionless",
        "The standard deviation of SCALE over 200 bootstrap resamples of "
        "this region and arm's own usable pixels. NaN where the region has "
        "too few usable pixels."),
    ("column-scale_twomass_region", "SLOPE_H_KS_PER_AK"): ("mag (H-Ks) per mag A_K",
        "For this region and the matching arm in ARM, the ordinary-least-"
        "squares slope of the background stars' median H minus Ks color "
        "against that arm's per-pixel map column, before the A_H/A_K "
        "conversion SCALE applies. NaN where the region has too few "
        "usable pixels."),
    ("column-scale_twomass_region", "INTERCEPT_H_KS"): ("mag",
        "For this region and the matching arm in ARM, the intercept of "
        "the ordinary-least-squares regression of the background "
        "stars' median H minus Ks color against that arm's per-pixel "
        "map column: the background stars' predicted median H minus Ks "
        "color at zero map column. NaN where the region has too few "
        "usable pixels."),
    ("column-scale_twomass_region", "N_STARS"): ("stars",
        "For this region and the matching arm in ARM, how many background "
        "2MASS stars, summed over the pixels usable for that arm's "
        "regression, went into it."),
    ("column-scale_twomass_region", "N_PIXELS"): ("pixels",
        "For this region and the matching arm in ARM, how many nside-512 "
        "pixels carried both a usable background-star color and a usable "
        "arm column, and so went into the regression."),

    # catalog.depths -- depths_sesna_region: the fitted 50%-completeness
    # detection curve per region and band.
    ("depths_sesna_region", "REGION"): ("region name",
        "The name of the region this row describes, in the fixed thirty-"
        "region order every region-axis product in this package shares."),
    ("depths_sesna_region", "DELTA_DEX"): ("dex",
        "For each of the five Spitzer bands (I1, I2, I3, I4, M1, this "
        "dataset's own column order), the fitted offset from this region's "
        "90%-completeness map value to its 50%-completeness flux: "
        "F_lim,50 = DCOMP90_MJY * 10**(-DELTA_DEX). A column can be "
        "negative where a band's detections turn over brighter than the "
        "map's own 90% level."),
    ("depths_sesna_region", "SIGMA_DELTA_DEX"): ("dex",
        "The 1-sigma fit uncertainty on DELTA_DEX, same five Spitzer bands "
        "and column order."),
    ("depths_sesna_region", "W_MAG"): ("mag",
        "For each of the five Spitzer bands (I1, I2, I3, I4, M1, this "
        "dataset's own column order), the fitted roll-off width of this "
        "region's detection curve, in magnitude. WIDTH_DEX in this file is "
        "this value times 0.4 (dex per magnitude)."),
    ("depths_sesna_region", "FIT_RESIDUAL"): ("dimensionless",
        "For each of the five Spitzer bands (I1, I2, I3, I4, M1, this "
        "dataset's own column order), the fitted detection curve's "
        "relative L1 distance from this region's own observed histogram, "
        "measured only past the roll-off's own 90% point."),
    ("depths_sesna_region", "F50_2MASS_MJY"): ("mJy",
        "For each of the three 2MASS bands (J, H, Ks, this dataset's own "
        "column order), this region's fitted 50%-completeness flux."),
    ("depths_sesna_region", "FIT_RESIDUAL_2MASS"): ("dimensionless",
        "For each of the three 2MASS bands (J, H, Ks, this dataset's own "
        "column order), the fitted detection curve's relative L1 distance "
        "from this region's own observed magnitude histogram, measured "
        "only past one magnitude brighter than the fitted 50% point."),
    ("depths_sesna_region", "SUBSTITUTED"): ("boolean",
        "For each of the eight bands (J, H, Ks, I1, I2, I3, I4, M1, this "
        "dataset's own column order), True where this region's own fit did "
        "not converge and DELTA_DEX and WIDTH_DEX for that band were "
        "replaced by that band's median over the regions whose fit did "
        "converge. Always False for the three 2MASS bands, which carry no "
        "convergence check."),
    ("depths_sesna_region", "WIDTH_DEX"): ("dex",
        "For each of the eight bands (J, H, Ks, I1, I2, I3, I4, M1, this "
        "dataset's own column order), this region's effective detection "
        "roll-off width: W_MAG times 0.4 for the five Spitzer bands, the "
        "directly fitted magnitude-space width for the three 2MASS bands."),
    ("depths_sesna_region", "SIGMA_WIDTH_DEX"): ("dex",
        "For each of the eight bands (J, H, Ks, I1, I2, I3, I4, M1, this "
        "dataset's own column order), the 1-sigma uncertainty on WIDTH_DEX "
        "for the five Spitzer bands. NaN for the three 2MASS bands, which "
        "carry no resampling uncertainty."),

    # catalog.coverage -- coverage_sesna_hpx512: the catalog's own observed
    # IRAC footprint on the nside-512 grid, per region.
    ("coverage_sesna_hpx512", "HPX_PIX"): ("nested HEALPix pixel, nside 512",
        "The region's admitted nside-512 pixel numbers (nested ordering), "
        "ascending. Row i here is row i of FRAC."),
    ("coverage_sesna_hpx512", "FRAC"): ("dimensionless",
        "For the matching pixel in HPX_PIX, the fraction of its 16 nside-2048 "
        "child pixels holding at least one SESNA source with a measured 3.6, "
        "4.5, 5.8 or 8.0 micron detection. This is the catalog's own observed "
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
        "One value per band (J, H, Ks, 3.6, 4.5, 5.8, 8.0 and 24 micron, this file's own band order), the "
        "roll-off width of the region's fitted detection curve: how many factors "
        "of 10 in flux the detection fraction takes to fall from high to low "
        "around F_50_REGION_MJY. The same eight values apply to every source in "
        "this file."),
    ("limits_sesna_source", "F_50_REGION_MJY"): ("mJy",
        "One value per band (J, H, Ks, 3.6, 4.5, 5.8, 8.0 and 24 micron, this file's own band order), the "
        "region's own fitted 50%-completeness flux: the flux at which half of "
        "the region's sources at that depth are detected. The same eight values "
        "apply to every source in this file. A value of +inf means the region "
        "did not sample enough of that band's roll-off to fit one."),
    ("limits_sesna_source", "ALPHA_REGION"): ("dimensionless",
        "One value per band (J, H, Ks, 3.6, 4.5, 5.8, 8.0 and 24 micron, this file's own band order), the "
        "power-law slope of the region's own source counts at fluxes well above "
        "F_50_REGION_MJY, from the region's own fitted detection curve. The "
        "same eight values apply to every source in this file."),
    ("limits_sesna_source", "DCOMP90_REF_LOG10"): ("log10 mJy",
        "One value per band (J, H, Ks, 3.6, 4.5, 5.8, 8.0 and 24 micron, this file's own band order), the "
        "log10 90%-completeness flux that F_LIM_50_MJY's per-source shift is "
        "measured from: the median, over the region's own low-extinction, "
        "well-covered sources, of the log10 90%-completeness flux each of "
        "those sources carries in the curated catalog. The same eight "
        "values apply to every source in this file."),
    ("limits_sesna_source", "LIMIT_KIND"): ("code",
        "One value per band (J, H, Ks, 3.6, 4.5, 5.8, 8.0 and 24 micron, this file's own band order), how "
        "F_50_REGION_MJY for that band was obtained: \"fit\" means the region's "
        "own source counts show a genuine roll-off; \"bound\" means this band's "
        "faint end is set by a different, more restrictive band's requirement, "
        "so F_50_REGION_MJY is a bound inherited from that other band rather "
        "than this band's own turnover; \"unsurveyed\" means the region has too "
        "few detections in that band to fit a roll-off at all, and "
        "F_50_REGION_MJY is +inf. The same eight values apply to every source "
        "in this file."),

    # catalog.depth_grid, per-pixel product -- depth-grid_sesna_hpx512: the
    # same per-source limits, summarized on the nside-512 grid.
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
        "50%-completeness flux limit (the catalog's own per-source limits product). A pixel with no sources of its own (N_SOURCES = 0) takes its nearest occupied pixel's value."),
    ("depth-grid_sesna_hpx512", "F_LIM_50_PIX_MJY"): ("mJy",
        "For the matching pixel in HPX_PIX_512 and each of the eight bands (J, "
        "H, Ks, 3.6, 4.5, 5.8, 8.0 and 24 micron, this file's own band order), "
        "the median, over the pixel's own sources, of each source's own "
        "50%-completeness flux limit (the catalog's own per-source limits product). A pixel with no sources of its own (N_SOURCES = 0) takes its nearest occupied pixel's value."),
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
        "Which field this object was cataloged in: Perseus or Barnard 1."),
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

    ("uwish2_knots_survey", "UWISH2_ID"): ("UWISH2 catalog identifier",
        "This feature's identifier in the UWISH2 catalog (Froebrich et al. "
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
        "Right ascension of the image center, equinox J2000."),
    ("uwish2_images_knots_survey", "DEC_DEG"): ("deg",
        "Declination of the image center, equinox J2000."),
    ("uwish2_images_knots_survey", "GLON_DEG"): ("deg",
        "Galactic longitude of the image center."),
    ("uwish2_images_knots_survey", "GLAT_DEG"): ("deg",
        "Galactic latitude of the image center."),
    ("uwish2_images_knots_survey", "NOISE"): ("counts",
        "The image's own one-pixel background noise level, in the map's "
        "native calibrated counts (Froebrich et al. 2015, their Table C1)."),

    # sky.derived.trilegal_colour -- colour_trilegal_survey: TRILEGAL's own
    # Gaia G - 2MASS Ks color as a function of atmosphere alone.
    ("colour_trilegal_survey", "LOG_TEFF_EDGES"): ("dex log10(K)",
        "The edges of this file's log10(effective temperature) bins, "
        "0.02 dex wide: bin i runs from LOG_TEFF_EDGES[i] to "
        "LOG_TEFF_EDGES[i+1]."),
    ("colour_trilegal_survey", "LOG_G_EDGES"): ("dex log10(cm/s^2)",
        "The edges of this file's log10(surface gravity) bins, 0.25 dex "
        "wide: bin j runs from LOG_G_EDGES[j] to LOG_G_EDGES[j+1]."),
    ("colour_trilegal_survey", "MH_EDGES"): ("dex [M/H]",
        "The edges of this file's metallicity bins, 0.25 dex wide: bin k "
        "runs from MH_EDGES[k] to MH_EDGES[k+1]."),
    ("colour_trilegal_survey", "G_MINUS_KS_MEDIAN"): ("mag",
        "For the cell at (LOG_TEFF_EDGES[i], LOG_G_EDGES[j], MH_EDGES[k]), "
        "the median Gaia G minus 2MASS Ks color of the TRILEGAL stars that "
        "fall in it. NaN where COUNT is 0."),
    ("colour_trilegal_survey", "G_MINUS_KS_HALFWIDTH"): ("mag",
        "For the cell at (LOG_TEFF_EDGES[i], LOG_G_EDGES[j], MH_EDGES[k]), "
        "half the difference between the 84th and 16th percentile of "
        "the Gaia G minus 2MASS Ks colors of the TRILEGAL stars that "
        "fall in it: a robust one-sided spread around the median. NaN "
        "where COUNT is 0."),
    ("colour_trilegal_survey", "COUNT"): ("stars",
        "For the cell at (LOG_TEFF_EDGES[i], LOG_G_EDGES[j], MH_EDGES[k]), "
        "how many TRILEGAL stars from the download fall in it."),

    # sky.derived.planck_source_column -- column_planck_source: the Planck
    # thermal-dust arm of the per-source extinction column.
    ("column_planck_source", "A_K"): ("mag A_K",
        "For every cataloged source of this region, the K-band extinction "
        "from the Planck R1.20 thermal-dust optical depth (TAU353) "
        "bilinearly interpolated at the source's own Galactic position."),
    ("column_planck_source", "SIGMA_A_K"): ("mag A_K",
        "The total 1-sigma uncertainty on A_K, combining SIGMA_STAT_K, "
        "SIGMA_WITHIN_K and SIGMA_REGION_K in quadrature."),
    ("column_planck_source", "SIGMA_STAT_K"): ("mag A_K",
        "The part of A_K's uncertainty from the Planck map's own per-pixel "
        "statistical error on TAU353 at the source's position."),
    ("column_planck_source", "SIGMA_WITHIN_K"): ("mag A_K",
        "The part of A_K's uncertainty from the measured scatter, within "
        "this source's region, between the Planck-based column and the "
        "reference column it is calibrated against: a fixed offset plus a "
        "term growing with A_K."),
    ("column_planck_source", "SIGMA_REGION_K"): ("mag A_K",
        "The part of A_K's uncertainty from the region-to-region scatter "
        "of the Planck-based column against the reference column it is "
        "calibrated against, not reduced by beam averaging since a "
        "single sightline sees exactly one Planck beam."),

    # sky.derived.edenhofer_samples -- profile-sigma-samples_edenhofer_sightline:
    # the across-sample uncertainty on the Edenhofer et al. (2024) extinction
    # profile, from its 12 released posterior samples.
    ("profile-sigma-samples_edenhofer_sightline", "HPX_PIX_256"): ("nested HEALPix pixel, nside 256",
        "This region's own nside-256 pixel numbers (nested ordering). Row i "
        "here is row i of SIGMA_SAMPLES_K and SIGMA_RATIO_SAMPLES."),
    ("profile-sigma-samples_edenhofer_sightline", "DIST_PC"): ("pc",
        "The distance grid along the line of sight that SIGMA_SAMPLES_K and "
        "SIGMA_RATIO_SAMPLES are tabulated on, shared by every pixel in "
        "this file."),
    ("profile-sigma-samples_edenhofer_sightline", "SIGMA_SAMPLES_K"): ("mag A_K",
        "For the matching pixel in HPX_PIX_256 and each distance in "
        "DIST_PC, the standard deviation, across the map's 12 released "
        "posterior samples, of the cumulative K-band extinction to that "
        "distance. This is the sample-based uncertainty on the cumulative "
        "column, in place of a correlated- or uncorrelated-sum bound."),
    ("profile-sigma-samples_edenhofer_sightline", "SIGMA_RATIO_SAMPLES"): ("dimensionless",
        "For the matching pixel in HPX_PIX_256 and each distance in "
        "DIST_PC, the standard deviation, across the map's 12 released "
        "posterior samples, of the ratio of the cumulative extinction at "
        "that distance to the sample's own total extinction along the "
        "whole sightline. A star's own position along the sightline is "
        "set by this same ratio, so this is the sample uncertainty on "
        "that positioning quantity: smaller near the far end (where the "
        "ratio is pinned near 1 for every sample) than SIGMA_SAMPLES_K's "
        "absolute uncertainty would suggest."),

    # sky.derived.swire_galaxies -- galaxies_swire_survey: the surviving
    # SWIRE galaxies, survey-wide, their IRAC colors and flux-grid node.
    ("galaxies_swire_survey", "LOG10_S"): ("log10 mJy",
        "This galaxy's 4.5 micron (IRAC I2) flux density: 10**x is a flux in "
        "mJy."),
    ("galaxies_swire_survey", "NODE"): ("grid node index",
        "Which point of LOG10_S_GRID this galaxy's LOG10_S falls nearest to "
        "(the midpoint between two adjacent grid points marks the "
        "boundary), or -1 where LOG10_S falls outside the grid's own range."),
    ("galaxies_swire_survey", "COLOUR_I1I2"): ("dex",
        "log10(3.6 micron flux) minus log10(4.5 micron flux), both flux "
        "densities in mJy, for this surviving SWIRE galaxy. NaN where "
        "either band's flux or uncertainty is missing or not positive."),
    ("galaxies_swire_survey", "COLOUR_I2I3"): ("dex",
        "log10(4.5 micron flux) minus log10(5.8 micron flux), both flux "
        "densities in mJy, for this surviving SWIRE galaxy. NaN where "
        "either band's flux or uncertainty is missing or not positive."),
    ("galaxies_swire_survey", "COLOUR_I2I4"): ("dex",
        "log10(4.5 micron flux) minus log10(8.0 micron flux), both flux "
        "densities in mJy, for this surviving SWIRE galaxy. NaN where "
        "either band's flux or uncertainty is missing or not positive."),
    ("galaxies_swire_survey", "SIGMA_COLOUR_I1I2"): ("dex",
        "The 1-sigma uncertainty on COLOUR_I1I2: each band's flux "
        "uncertainty divided by its own flux and by ln(10), combined in "
        "quadrature over the 3.6 and 4.5 micron bands. NaN where either "
        "band's flux or uncertainty is missing or not positive."),
    ("galaxies_swire_survey", "SIGMA_COLOUR_I2I3"): ("dex",
        "The 1-sigma uncertainty on COLOUR_I2I3: each band's flux "
        "uncertainty divided by its own flux and by ln(10), combined in "
        "quadrature over the 4.5 and 5.8 micron bands. NaN where either "
        "band's flux or uncertainty is missing or not positive."),
    ("galaxies_swire_survey", "SIGMA_COLOUR_I2I4"): ("dex",
        "The 1-sigma uncertainty on COLOUR_I2I4: each band's flux "
        "uncertainty divided by its own flux and by ln(10), combined in "
        "quadrature over the 4.5 and 8.0 micron bands. NaN where either "
        "band's flux or uncertainty is missing or not positive."),
    ("galaxies_swire_survey", "FIELD"): ("field index",
        "Which of the six SWIRE fields this galaxy comes from, as a "
        "position in this file's own FIELDS attribute (a semicolon-"
        "separated list of the six field file names): 0 the first field "
        "named there, and so on."),
    ("galaxies_swire_survey", "LOG10_S_GRID"): ("log10 mJy",
        "The 61 flux-grid points every SWIRE-galaxy product shares, from "
        "SWIRE's own 5-sigma depth at 4.5 micron to the brightest "
        "tabulated point in Fazio et al. (2004)'s galaxy counts law. 10**x "
        "is a flux in mJy."),
    ("galaxies_swire_survey", "N_NODE_I1I2"): ("galaxies",
        "For each of the 61 points in LOG10_S_GRID, how many surviving "
        "galaxies land nearest that point (NODE) and also carry a finite "
        "COLOUR_I1I2."),
    ("galaxies_swire_survey", "N_NODE_ALL"): ("galaxies",
        "For each of the 61 points in LOG10_S_GRID, how many surviving "
        "galaxies land nearest that point (NODE) and also carry a finite "
        "value in all three colors, COLOUR_I1I2, COLOUR_I2I3 and "
        "COLOUR_I2I4."),

    # sky.derived.juvela_extinction -- extinction_juvela_source (per
    # region) and extinction_juvela_sightline (survey-wide): the Juvela &
    # Montillaud (2016) NICEST star-color extinction map, converted to
    # A_K.
    ("extinction_juvela_source", "A_K"): ("mag A_K",
        "For every cataloged source of this region, the NICEST star-color "
        "extinction map's value at the source's own sky position, converted "
        "from the map's native A_J to A_K with the adopted diffuse "
        "extinction law's own A_J/A_K ratio."),
    ("extinction_juvela_source", "HPX_PIX_1024"): ("nested HEALPix pixel, nside 1024",
        "For every cataloged source of this region, the nested nside-1024 "
        "pixel number of the source's own sky position: the map cell this "
        "file's A_K value is read at."),
    ("extinction_juvela_sightline", "HPX_PIX_256"): ("nested HEALPix pixel, nside 256",
        "The admitted nside-256 pixel numbers of every one of the thirty "
        "regions, nested ordering. Row i here is row i of A_K."),
    ("extinction_juvela_sightline", "A_K"): ("mag A_K",
        "For the matching pixel in HPX_PIX_256, the mean of the NICEST "
        "star-color extinction map's A_K value over all 64 of that pixel's "
        "nside-2048 children, covering the whole sightline rather than only "
        "where a source happens to fall."),

    # sky.derived.gaia_twomass_counts -- joint-counts_gaia-twomass_hpx512:
    # the joint Gaia G x 2MASS Ks anchor count, per region.
    ("joint-counts_gaia-twomass_hpx512", "HPX_PIX_512"): ("nested HEALPix pixel, nside 512",
        "The region's own occupied nside-512 pixel numbers (nested ordering), "
        "ascending. Row i here is row i of N_GK."),
    ("joint-counts_gaia-twomass_hpx512", "G_EDGES"): ("mag Gaia G",
        "The edges of the 1-magnitude Gaia G bins this file counts stars in, "
        "10 to 19: bin j runs from G_EDGES[j] to G_EDGES[j+1]."),
    ("joint-counts_gaia-twomass_hpx512", "KS_EDGES"): ("mag 2MASS Ks",
        "The edges of the half-magnitude 2MASS Ks bins this file counts "
        "stars in, 9.0 to 14.3: bin j runs from KS_EDGES[j] to KS_EDGES[j+1]."),
    ("joint-counts_gaia-twomass_hpx512", "N_GK"): ("stars",
        "For the matching pixel in HPX_PIX_512, each Gaia G bin in G_EDGES "
        "and each 2MASS Ks bin in KS_EDGES, how many Gaia-2MASS crossmatched "
        "stars fall in that pixel and that pair of bins."),

    # sky.derived.coverage -- coverage_spitzer_sightline (nside-256) and
    # coverage_spitzer_hpx512 (nside-512): the Spitzer mosaic coverage
    # fraction per pixel and band, per region.
    ("coverage_spitzer_sightline", "HPX_PIX"): ("nested HEALPix pixel, nside 256",
        "The region's nside-256 pixel numbers (nested ordering) that its own "
        "Spitzer mosaics touch, ascending. Row i here is row i of FRAC."),
    ("coverage_spitzer_sightline", "FRAC"): ("dimensionless",
        "For the matching pixel in HPX_PIX and each of the five Spitzer bands "
        "named in BANDS, the fraction of that nside-256 pixel's area that "
        "falls on a covered mosaic pixel in that band."),
    ("coverage_spitzer_sightline", "BANDS"): ("band key",
        "The five Spitzer band keys, in the column order FRAC's second axis "
        "uses: I1, I2, I3, I4 (IRAC), M1 (MIPS)."),
    ("coverage_spitzer_hpx512", "HPX_PIX"): ("nested HEALPix pixel, nside 512",
        "The region's nside-512 pixel numbers (nested ordering) that its own "
        "Spitzer mosaics touch, ascending. Row i here is row i of FRAC."),
    ("coverage_spitzer_hpx512", "FRAC"): ("dimensionless",
        "For the matching pixel in HPX_PIX and each of the five Spitzer bands "
        "named in BANDS, the fraction of that nside-512 pixel's area that "
        "falls on a covered mosaic pixel in that band."),
    ("coverage_spitzer_hpx512", "BANDS"): ("band key",
        "The five Spitzer band keys, in the column order FRAC's second axis "
        "uses: I1, I2, I3, I4 (IRAC), M1 (MIPS)."),

    # sky.derived.gaia_counts / twomass_counts / ukidss_counts -- anchor
    # star counts on the region's own nside-512 pixels, one magnitude band
    # per survey.
    ("counts_gaia_hpx512", "HPX_PIX_512"): ("nested HEALPix pixel, nside 512",
        "The region's own occupied nside-512 pixel numbers (nested ordering), "
        "ascending. Row i here is row i of N."),
    ("counts_gaia_hpx512", "MAG_EDGES"): ("mag Gaia G",
        "The edges of the eleven 1-magnitude Gaia G bins this file counts "
        "stars in, 10 to 21: bin j runs from MAG_EDGES[j] to MAG_EDGES[j+1]."),
    ("counts_gaia_hpx512", "N"): ("stars",
        "For the matching pixel in HPX_PIX_512 and each magnitude bin in "
        "MAG_EDGES, how many Gaia DR3 sources fall in that pixel and bin."),

    ("counts_twomass_hpx512", "HPX_PIX_512"): ("nested HEALPix pixel, nside 512",
        "The region's own occupied nside-512 pixel numbers (nested ordering), "
        "ascending. Row i here is row i of N."),
    ("counts_twomass_hpx512", "MAG_EDGES"): ("mag 2MASS Ks",
        "The edges of the half-magnitude 2MASS Ks bins this file counts "
        "stars in, 9.0 to 14.3: bin j runs from MAG_EDGES[j] to "
        "MAG_EDGES[j+1]."),
    ("counts_twomass_hpx512", "N"): ("stars",
        "For the matching pixel in HPX_PIX_512 and each magnitude bin in "
        "MAG_EDGES, how many 2MASS Point Source Catalog detections fall in "
        "that pixel and bin."),

    ("counts_ukidss_hpx512", "HPX_PIX_512"): ("nested HEALPix pixel, nside 512",
        "The region's own occupied nside-512 pixel numbers (nested ordering), "
        "ascending. Row i here is row i of N."),
    ("counts_ukidss_hpx512", "MAG_EDGES"): ("mag UKIRT K",
        "The edges of the half-magnitude UKIDSS GPS K bins (UKIRT "
        "photometric system) this file counts stars in, 9.0 to 17.0: bin j "
        "runs from MAG_EDGES[j] to MAG_EDGES[j+1]."),
    ("counts_ukidss_hpx512", "N"): ("stars",
        "For the matching pixel in HPX_PIX_512 and each magnitude bin in "
        "MAG_EDGES, how many clean-photometry UKIDSS GPS detections fall in "
        "that pixel and bin."),

    ("colours_knots_survey", "LOG10_RATIO"): ("dex",
        "log10(F_IRAC_band / F_2.12um) for every Giannini et al. (2013) knot "
        "with a measured, non-upper-limit flux in both the 2.12 micron line "
        "and this group's own IRAC band, one value per knot. This dataset's "
        "enclosing group is named by the Spitzer IRAC band: I1 3.6 micron, I2 "
        "4.5 micron, I3 5.8 micron, I4 8.0 micron."),

    # fittp.classify -- posterior_classification_source (P8): the class and
    # subclass posterior, the MAP class, and the imputed flux, one file per
    # region (moved from fittp/classify.py's own _READINGS, READINGS brief
    # section 3).
    ("posterior_classification_source", "NAME"): ("source name",
        "The source's name from the SESNA catalog. Rows are in catalog order."),
    ("posterior_classification_source", "CLASS_SESNA"): ("SESNA class code",
        "SESNA's own classification, copied from the catalog: 0 deeply embedded "
        "protostar, 1 class I, 2 class II, 3 transition disk, 9 H2 shock blob, 19 "
        "PAH emitter (star-forming galaxy), 29 AGN, 39 PAH-contaminated source, 49 "
        "generic galaxy, 99 diskless star, -100 unclassified. No other column here "
        "depends on it."),
    ("posterior_classification_source", "N_DETECTED"): ("bands",
        "How many of the eight bands (J, H, Ks, 3.6, 4.5, 5.8, 8.0 and 24 micron) "
        "have a measured, positive flux."),
    ("posterior_classification_source", "MAP_CLASS"): ("class position",
        "The most probable class, as a position in the CLASSES attribute, counting "
        "from zero. -1 means the source could not be fitted: fewer than two "
        "measured bands, or a flux error that is not finite. -2 means the fit "
        "itself was valid but no class has any prior mass at this source's fitted "
        "brightness and depth."),
    ("posterior_classification_source", "P_CLASS"): ("probability",
        "Probability of each class: field star, dusty evolved star, "
        "nebula-contaminated aperture, background galaxy, young stellar object, "
        "shocked gas knot. Column order is the CLASSES attribute. Sums to 1."),
    ("posterior_classification_source", "P_SUBCLASS"): ("probability",
        "Probability of each of 25 subdivisions, named class and subdivision in the "
        "SUBCLASSES attribute: field stars by spectral type O to T, evolved stars "
        "oxygen-rich or carbon-rich, galaxies active, star-forming, composite or "
        "passive, young stellar objects class 0 to III or transition disk, knots J, "
        "C, steady C or C-J shocks. A class's columns sum to its P_CLASS; all 25 "
        "sum to 1."),
    ("posterior_classification_source", "P_YSO"): ("probability",
        "Probability that the source is a young stellar object: the young stellar "
        "object column of P_CLASS."),
    ("posterior_classification_source", "ENTROPY_CLASS"): ("fraction of maximum",
        "How spread the six class probabilities are, divided by their maximum "
        "spread. 0 means one class takes all the probability, 1 means all six are "
        "equally likely."),
    ("posterior_classification_source", "ENTROPY_SUBCLASS"): ("fraction of maximum",
        "How spread the 25 subdivision probabilities are, divided by their maximum "
        "spread. 0 means one subdivision takes all the probability, 1 means all 25 "
        "are equally likely."),
    ("posterior_classification_source", "A_K_POST"): ("magnitudes of K-band extinction",
        "The fitted extinction in front of the source, one column per class in "
        "CLASSES order: the value if the source were of that class. Read the "
        "MAP_CLASS column. Each model's own fitted range of extinction is averaged "
        "over all models and both extinction laws, weighted by each model's share "
        "of the support for the source: its fit to the measured fluxes, how many "
        "such objects the sky holds at that position and brightness, and what Gaia "
        "says."),
    ("posterior_classification_source", "A_K_POST_SIG"): ("magnitudes of K-band extinction",
        "The standard deviation of that fitted extinction, one column per class. It "
        "covers both the precision of the fit and disagreement between the models "
        "that fit the source. It is not a formal fitting error."),
    ("posterior_classification_source", "LOG10_FLUX_IMPUTED"): ("log10 of flux in mJy",
        "The eight-band spectrum, band order J, H, Ks, 3.6, 4.5, 5.8, 8.0 and 24 "
        "micron. A measured band holds log10 of the catalog flux; the rest hold the "
        "support-weighted average over the models that fit the source. 10**x is a "
        "flux in mJy."),
    ("posterior_classification_source", "LOG10_FLUX_IMPUTED_COV"): ("squared dex",
        "Covariance of those eight values, one dex being a factor of 10. On an "
        "estimated band, s = sqrt(diagonal) is the uncertainty in log flux, so the "
        "flux spans 10**(x-s) to 10**(x+s) with 68 percent probability. On a "
        "measured band the diagonal is the catalog error, recovered in mJy as 10**x "
        "times 2.3026 times s. An entry between a measured and an estimated band is "
        "not zero and is needed for a color across the pair; between two measured "
        "bands it is zero."),
    ("posterior_classification_source", "LOG10_CANDIDATE_FLUX"): ("log10 of flux in mJy",
        "The same spectrum under each class in turn, CLASSES order then band order. "
        "A measured band holds log10 of the catalog flux; the rest hold that "
        "class's own estimate."),

    # fittp.classify -- sensitivity_classification_region (P9): the
    # literature-band sensitivity, one row per region (moved from fittp/
    # classify.py's own inline map in write_sensitivity, READINGS brief
    # section 3).
    ("sensitivity_classification_region", "REGION"): ("region name",
        "The region this row describes."),
    ("sensitivity_classification_region", "RUN"): ("test name",
        "One of nine tests. Each changes how many objects of a class the sky "
        "is expected to hold, by as much as the published measurements allow, "
        "to see how far the classification moves."),
    ("sensitivity_classification_region", "SCALING"): ("factor",
        "The factor a test applied to each class's expected numbers, 1.0 "
        "where it leaves a class alone. Class order is the CLASSES "
        "attribute. The young-star floor test holds that region's average "
        "factor over its own sources, not a single published number."),
    ("sensitivity_classification_region", "FRAC_MAP_CHANGED"): ("fraction",
        "The fraction of the region's sources whose most "
        "probable class changed under that test."),
    ("sensitivity_classification_region", "N_PYSO_ABOVE_HALF"): ("sources",
        "How many sources have a young stellar object "
        "probability above one half under each test. The first "
        "column is the reported classification, untested."),
    ("sensitivity_classification_region", "N_SOURCES"): ("sources", "How many sources the region holds."),

    # fittp.cascade -- cascade_classification_source (P10): the Gutermuth
    # color-cascade verdicts, measured and imputed halves, one file per
    # region (moved from fittp/cascade.py's own _READINGS, READINGS brief
    # section 3).
    ("cascade_classification_source", "NAME"): ("source name",
        "The source's name from the SESNA catalog. Rows are in catalog order."),
    ("cascade_classification_source", "N_DETECTED"): ("bands",
        "How many of the eight bands (J, H, Ks, 3.6, 4.5, 5.8, 8.0 and 24 micron) "
        "have a measured, positive flux."),
    ("cascade_classification_source", "VERDICT_MEASURED"): ("SESNA class code",
        "The class the Gutermuth et al. (2009) color cuts give from the measured "
        "fluxes alone. The cuts are deterministic, so this is the category the "
        "colors fall in, not a most likely class. Codes: 0 deeply embedded "
        "protostar, 1 class I, 2 class II, 3 transition disk, 9 H2 shock blob, 19 "
        "PAH emitter (star-forming galaxy), 29 AGN, 39 PAH-contaminated source, 49 "
        "generic galaxy, 99 diskless star, -100 unclassified, with -100 where no "
        "cut applies."),
    ("cascade_classification_source", "VERDICT_IMPUTED"): ("SESNA class code",
        "The same color cuts after the unmeasured bands are filled in with the "
        "pipeline's estimates. Those estimates assume the class the pipeline chose, "
        "so this is a comparison, not independent evidence. Codes as in "
        "VERDICT_MEASURED."),
    ("cascade_classification_source", "P_VERDICT_MEASURED"): ("probability",
        "Probability of each of the color cuts' eleven categories when the measured "
        "fluxes are varied within their errors. Column order is the LABELS "
        "attribute. Sums to 1; the unclassified column is the chance that no cut "
        "applies."),
    ("cascade_classification_source", "PSI_VOTES"): ("votes",
        "Four indicators, each casting one vote across the six classes in CLASSES "
        "order: the class mix expected at the source's position, Gaia's detection "
        "and parallax, the Gutermuth color cuts on the measured fluxes, and those "
        "cuts on the filled-in spectrum. An indicator with nothing to say abstains, "
        "so a row sums to between 0 and 4. Reported only; the classification does "
        "not use it."),
    ("cascade_classification_source", "ENTROPY_PSI_VOTES"): ("fraction of maximum",
        "How spread those votes are, divided by their maximum spread. 0 means the "
        "indicators that voted agreed, 1 means they spread evenly over the six "
        "classes. Not a number where none voted."),

    # fittp.library_resolution -- library-resolution_check_survey: whether
    # each class's library has enough templates to resolve the survey's own
    # photometric error, one survey-wide file (moved from fittp/
    # library_resolution.py's own _READINGS, READINGS brief section 3).
    ("library-resolution_check_survey", "LIBRARY"): ("library name",
        "The model library this row describes, one per class."),
    ("library-resolution_check_survey", "SIGMA_LIB_DEX"): ("dex",
        "How finely that library samples spectral shape: the typical distance from "
        "a model to its nearest neighbor. The fit adds this to each band's "
        "measurement error, so no model can beat a near-identical one by more than "
        "the sampling allows."),
    ("library-resolution_check_survey", "SIGMA_I"): ("dex",
        "The survey's flux uncertainty per band, as the tenth percentile over "
        "sources. Band order is the BANDS dataset."),
    ("library-resolution_check_survey", "BANDS"): ("band name",
        "The eight SESNA bands, in SIGMA_I's column order."),
    ("library-resolution_check_survey", "OTHER_LIBRARY"): ("library name",
        "A library other than the young stellar object one."),
    ("library-resolution_check_survey", "OTHER_N_TEMPLATES"): ("models",
        "How many models that library holds."),
    ("library-resolution_check_survey", "OTHER_D50"): ("dex",
        "Median distance from a model of that library to its nearest neighbor."),
    ("library-resolution_check_survey", "OTHER_D90"): ("dex",
        "The same distance at the ninetieth percentile: the spacing of that "
        "library's most isolated models."),
    ("library-resolution_check_survey", "OTHER_THICK_ENOUGH"): ("true or false",
        "Whether that library's models are spaced more widely than this check's "
        "tolerance, so adding models would sharpen the fit."),
    ("library-resolution_check_survey", "YSO_GROUP"): ("model group",
        "Which group of young stellar object models this row describes. The library "
        "is built in groups by evolutionary stage and geometry."),
    ("library-resolution_check_survey", "YSO_N_AT_SIZE"): ("models",
        "How many models that group holds at each of four sizes, thinned from the "
        "full group, so spacing can be measured against library size."),
    ("library-resolution_check_survey", "YSO_D50_AT_SIZE"): ("dex",
        "Median nearest-neighbor distance for that group at each of the four sizes."),
    ("library-resolution_check_survey", "YSO_D90_AT_SIZE"): ("dex",
        "Ninetieth-percentile nearest-neighbor distance at each of the four sizes."),
    ("library-resolution_check_survey", "YSO_D_EFF"): ("dimensionless",
        "How fast that group's spacing shrinks as models are added, from the slope "
        "of spacing against size. It acts like the number of dimensions the group "
        "really fills."),
    ("library-resolution_check_survey", "YSO_N_STAR"): ("models",
        "How many models that group would need to reach this check's tolerance, "
        "read off the fitted slope, at each of three tolerances."),

}

# --- fittp.sweep -- <CLS>_fit_source (P7): one file per {region, class},
# the class evidence, imputed flux and top-K posterior record. The same
# dataset names and readings repeat in every one of the six classes' own
# files (moved from fittp/sweep.py's own _READINGS, READINGS brief section
# 3), so they are filled once here and copied to each class's own stem --
# `fittp.sweep`'s own STEM, computed the same way `config.product_path`
# would (quantity=class code, source="fit", granule="source"). ---
_SWEEP_FIT_READINGS = {
    "NAME": ("source name",
        "The source's name from the SESNA catalog. Rows are in catalog order."),
    "N_DETECTED": ("bands",
        "How many of the eight bands (J, H, Ks, 3.6, 4.5, 5.8, 8.0 and 24 micron) "
        "have a measured, positive flux."),
    "LN_EVIDENCE": ("nats",
        "How well this class explains the source, one value per subdivision named "
        "in the SUBCLASSES attribute, as a natural logarithm. Each gathers every "
        "model of that subdivision under both extinction laws. Minus infinity means "
        "the subdivision has no models, or the class cannot explain the source, or "
        "the source could not be fitted."),
    "LOG10_FLUX_MEAN": ("log10 of flux in mJy",
        "This class's estimate of the eight-band spectrum, band order J, H, Ks, "
        "3.6, 4.5, 5.8, 8.0 and 24 micron, averaging its models by their share of "
        "the support for the source. 10**x is a flux in mJy."),
    "LOG10_FLUX_COV": ("squared dex",
        "Covariance of those eight values, one dex being a factor of 10. It "
        "combines disagreement between this class's models with the precision of "
        "the fitted brightness and extinction. sqrt(diagonal) is the uncertainty in "
        "log flux."),
    "A_K_POST": ("magnitudes of K-band extinction",
        "The fitted extinction in front of the source if it belongs to this class, "
        "averaging each model's own fitted range over all models and both "
        "extinction laws, weighted by each model's share of the support."),
    "A_K_POST_SIG": ("magnitudes of K-band extinction",
        "The standard deviation of that fitted extinction. It covers both the "
        "precision of the fit and disagreement between models, and is not a formal "
        "fitting error."),
    "P_DENSE": ("fraction",
        "How much of this class's support comes from the dense extinction law "
        "rather than the diffuse one. 0 where the sightline carries no dense dust "
        "in front of the source."),
    "OCCAM_GAP": ("nats",
        "How much more room this class's model library has near its single best "
        "model, as a natural logarithm, counting the models alone, not how well "
        "any of them fits this source. Never negative. A large value means many "
        "models occupy that neighborhood, a library-volume effect rather than "
        "evidence about the source."),
    "FRAC_CLAMPED": ("fraction",
        "The support-weighted fraction of this class's models whose best fit called "
        "for negative extinction and was held at zero."),
    "FAILED_ROWS": ("source position",
        "Positions, counting from zero in catalog order, of sources this run could "
        "not fit. Empty where every source was fitted."),
    "TOPK_MODEL": ("model position",
        "The five models of this class that best explain the source, best first, as "
        "positions in its model library. -1 where fewer were kept or the source "
        "could not be fitted. The other TOPK columns follow this same order."),
    "TOPK_FLUX": ("mJy",
        "Each of those five models' eight-band spectrum, band order J, H, Ks, 3.6, "
        "4.5, 5.8, 8.0 and 24 micron, at its own fitted brightness and extinction."),
    "TOPK_A_K": ("magnitudes of K-band extinction",
        "The extinction fitted for each of those five models. A model whose best "
        "fit called for negative extinction is held at zero."),
    "TOPK_LOG10_B": ("log10 of a scale factor",
        "The brightness fitted for each of those five models, as log10 of the "
        "factor multiplying the model's own reference spectrum."),
    "TOPK_CHI2": ("chi-squared",
        "Chi-squared of each of those five models against the measured fluxes, at "
        "its fitted brightness and extinction."),
    "TOPK_LN_L": ("nats",
        "How well each of those five models matches the photometry at its own "
        "best-fit brightness and extinction, as a natural logarithm: the fit to "
        "the measured fluxes, together with the chance the survey would have "
        "missed the flux predicted in each undetected band at that one point. "
        "The class evidence itself reads that same chance across a range of "
        "extinction, not at this one point, so it need not match this column."),
    "TOPK_LN_PRIOR_FIT": ("nats",
        "The prior and the fit to the measured bands integrated together over "
        "extinction along the fit's ridge; not separable into the two under the "
        "integral."),
    "TOPK_LN_UNSEEN": ("nats",
        "What the bands observed but not detected contributed, as a log factor "
        "given the prior and the measured fit, integrated over extinction; reduces "
        "to the point value when the fit pins the extinction."),
    "TOPK_LN_GAIA": ("nats",
        "What the Gaia detection and parallax contributed, as a log factor given "
        "the rest, integrated over extinction."),
    "TOPK_LN_PRIOR_ML": ("nats",
        "The prior at the best-fit point; a guide to how TOPK_LN_PRIOR_FIT splits "
        "between the prior and the fit, not an addend."),
    "TOPK_LN_GAMMA_ML": ("nats",
        "The Gaia term at the best-fit point; report-only; the vote reading uses "
        "it."),
    "TOPK_LAW": ("law position",
        "Which extinction law fits each of those five models better: 0 diffuse, 1 "
        "dense."),
}
for _cls in ("STAR", "AGB", "PAHC", "GAL", "YSO", "H2S"):
    _stem = "%s_fit_source" % _cls
    for _name, _rd in _SWEEP_FIT_READINGS.items():
        REGISTRY[(_stem, _name)] = _rd

# --- fittp.emission -- <lib>_emission_survey: the measured emission table
# E[k, b, v], one survey-wide file per library. The same dataset names and
# readings repeat in every one of the six libraries' own files (moved from
# fittp/emission.py's own _READINGS, READINGS brief section 3). ---
_EMISSION_READINGS = {
    "E": ("fraction",
        "For each kind of model in this library, the fraction that the color cuts "
        "of Gutermuth et al. (2009) place in each of their categories, as a "
        "function of how bright the model appears. The three axes are the kinds of "
        "model named in SUBCLASSES, the brightness bins bounded by LOG10_F45_EDGES, "
        "and the categories named in LABELS. Each kind and brightness adds to 1 "
        "across the categories."),
    "SUBCLASSES": ("model kind",
        "The kinds of model this library holds, in the row order E uses."),
    "LABELS": ("category name",
        "The eleven categories the color cuts of Gutermuth et al. (2009) can "
        "assign, in the column order E uses."),
    "LOG10_F45_EDGES": ("log10 of flux in mJy",
        "The edges of the brightness bins E uses, as the base-10 logarithm of "
        "apparent 4.5 micron flux in mJy. There is one more edge than there are "
        "bins."),
}
for _lib in ("sps", "agb", "pahc", "galz", "yso", "h2shock"):
    _stem = "%s_emission_survey" % _lib
    for _name, _rd in _EMISSION_READINGS.items():
        REGISTRY[(_stem, _name)] = _rd
