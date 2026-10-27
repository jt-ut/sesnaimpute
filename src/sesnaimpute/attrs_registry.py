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

    # sky.download.spitzer_coverage retired (COMPLETENESS_2026-10-20.md rule
    # 4): sky.derived.coverage now reads the completeness fields' own on-map
    # footprint (catalog.completeness.field_maps) directly, so no native
    # Spitzer mosaic mask file is acquired or registered any more.

    # sky.download.trilegal -- trilegal_ascii: one entry per TRILEGAL ascii
    # column this project saves (Girardi et al. 2005, A&A 436, 895), shared
    # by every converted part/pointing file (`build.fetch_region_part`,
    # `sky.download.trilegal.convert`) and the one Gaia+Tycho2+2MASS colour
    # query (`sky.download.trilegal.colour`); dataset names are the header
    # token with brackets dropped and '/' replaced by '_'
    # (`build.sanitize_column_name`), the verbatim token recovered from the
    # file's own HEADER attr. T6, 2026-10-19.
    ("trilegal_ascii", "Gc"): ("TRILEGAL component code",
        "TRILEGAL's own code for the Galactic component (thin disk, thick disk, "
        "bulge, halo, ...) this synthetic star was drawn from."),
    ("trilegal_ascii", "logAge"): ("log10 yr",
        "This synthetic star's age: 10**x is an age in years."),
    ("trilegal_ascii", "M_H"): ("dex [M/H]",
        "This synthetic star's metallicity, [M/H]."),
    ("trilegal_ascii", "m_ini"): ("Msun",
        "This synthetic star's initial (birth) mass."),
    ("trilegal_ascii", "logL"): ("log10 Lsun",
        "This synthetic star's bolometric luminosity: 10**x is a luminosity in "
        "solar luminosities."),
    ("trilegal_ascii", "logTe"): ("log10 K",
        "This synthetic star's effective temperature: 10**x is a temperature in "
        "Kelvin."),
    ("trilegal_ascii", "logg"): ("log10(cm/s^2)",
        "This synthetic star's surface gravity: 10**x is a gravity in cm/s^2."),
    ("trilegal_ascii", "m-M0"): ("mag",
        "This synthetic star's true distance modulus, dust-free (the query's own "
        "internal_extinction_kind = 0): TRILEGAL's own simulated grid value, not "
        "yet the continuous distance a reader draws from it "
        "(`population.field_stars.distance_pc`)."),
    ("trilegal_ascii", "Av"): ("mag",
        "The internal (circumstellar/disk) extinction TRILEGAL applied to this "
        "star: always 0, by this project's own query setting "
        "(internal_extinction_kind = 0, module docstring) -- the census applies "
        "its own measured line-of-sight extinction later instead."),
    ("trilegal_ascii", "m2_m1"): ("mass ratio",
        "This synthetic star's binary companion-to-primary mass ratio, m2/m1; 0 "
        "for a single star."),
    ("trilegal_ascii", "mbol"): ("mag bolometric",
        "This synthetic star's intrinsic bolometric apparent magnitude."),
    ("trilegal_ascii", "J"): ("mag 2MASS J Vega",
        "This synthetic star's intrinsic 2MASS J apparent magnitude."),
    ("trilegal_ascii", "H"): ("mag 2MASS H Vega",
        "This synthetic star's intrinsic 2MASS H apparent magnitude."),
    ("trilegal_ascii", "Ks"): ("mag 2MASS Ks Vega",
        "This synthetic star's intrinsic 2MASS Ks apparent magnitude."),
    ("trilegal_ascii", "3.6"): ("mag Spitzer IRAC I1 Vega",
        "This synthetic star's intrinsic Spitzer IRAC 3.6 micron apparent magnitude."),
    ("trilegal_ascii", "4.5"): ("mag Spitzer IRAC I2 Vega",
        "This synthetic star's intrinsic Spitzer IRAC 4.5 micron apparent magnitude."),
    ("trilegal_ascii", "5.8"): ("mag Spitzer IRAC I3 Vega",
        "This synthetic star's intrinsic Spitzer IRAC 5.8 micron apparent magnitude."),
    ("trilegal_ascii", "8.0"): ("mag Spitzer IRAC I4 Vega",
        "This synthetic star's intrinsic Spitzer IRAC 8.0 micron apparent magnitude."),
    ("trilegal_ascii", "24"): ("mag Spitzer MIPS M1 Vega",
        "This synthetic star's intrinsic Spitzer MIPS 24 micron apparent magnitude."),
    ("trilegal_ascii", "70"): ("mag Spitzer MIPS M2 Vega",
        "This synthetic star's intrinsic Spitzer MIPS 70 micron apparent magnitude "
        "(unused downstream of this file: the survey's own bands stop at 24 "
        "micron, module docstring)."),
    ("trilegal_ascii", "160"): ("mag Spitzer MIPS M3 Vega",
        "This synthetic star's intrinsic Spitzer MIPS 160 micron apparent "
        "magnitude (unused downstream of this file: the survey's own bands stop "
        "at 24 micron, module docstring)."),
    ("trilegal_ascii", "Mact"): ("Msun",
        "This synthetic star's current (actual, post-evolution) mass."),
    ("trilegal_ascii", "G"): ("mag Gaia DR2 G Vega",
        "This synthetic star's intrinsic Gaia DR2 G apparent magnitude (the one "
        "Gaia+Tycho2+2MASS colour query, `sky.download.trilegal.colour`)."),
    ("trilegal_ascii", "G_BP"): ("mag Gaia DR2 BP Vega",
        "This synthetic star's intrinsic Gaia DR2 BP apparent magnitude (the one "
        "Gaia+Tycho2+2MASS colour query, `sky.download.trilegal.colour`)."),
    ("trilegal_ascii", "G_RP"): ("mag Gaia DR2 RP Vega",
        "This synthetic star's intrinsic Gaia DR2 RP apparent magnitude (the one "
        "Gaia+Tycho2+2MASS colour query, `sky.download.trilegal.colour`)."),
    ("trilegal_ascii", "B_T"): ("mag Tycho-2 B_T Vega",
        "This synthetic star's intrinsic Tycho-2 B_T apparent magnitude (the one "
        "Gaia+Tycho2+2MASS colour query, `sky.download.trilegal.colour`)."),
    ("trilegal_ascii", "V_T"): ("mag Tycho-2 V_T Vega",
        "This synthetic star's intrinsic Tycho-2 V_T apparent magnitude (the one "
        "Gaia+Tycho2+2MASS colour query, `sky.download.trilegal.colour`)."),

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
        "90%-completeness flux read off Rob Gutermuth's measured completeness map "
        "at the source's nearest pixel (`catalog.completeness.lookup`), or NaN "
        "where the source's own pixel is off every field's map. ORIGIN_FNU in "
        "this file says which bands were filled and how."),
    ("sources_sesna_source", "SIGMA_FNU_MJY"): ("mJy",
        "The 1-sigma flux uncertainty for a measured detection in FNU_MJY, on the "
        "same mJy scale. Where FNU_MJY was filled instead of measured (see "
        "ORIGIN_FNU), this is the fixed value 0.99 (a flag the SED fitter reads as "
        "an upper limit, not a real measurement uncertainty), or NaN where "
        "ORIGIN_FNU is 91 (off-map, no fill at all)."),
    ("sources_sesna_source", "DCOMP90_MJY"): ("mJy",
        "The flux a source at this position and band would need to be detected at "
        "90% completeness: the survey's fixed bound for the 2MASS bands (J, H, "
        "Ks), or, for the three IRAC bands and the one MIPS band, Rob Gutermuth's "
        "measured completeness map value at the source's nearest pixel "
        "(`catalog.completeness.lookup`), NaN where the source's own pixel is off "
        "every field's map for that band."),
    ("sources_sesna_source", "ORIGIN_FNU"): ("code",
        "How FNU_MJY and DCOMP90_MJY in each band were obtained, one integer per "
        "band: 1 a measured detection, 2 the fixed 2MASS survey detection bound, "
        "90 covered but undetected (FNU_MJY filled from the completeness map's "
        "own 90%-completeness flux at the source's pixel), 91 off-map -- no "
        "completeness value and no non-detection term (COMPLETENESS_2026-10-20.md "
        "rule 2: \"sources with a catalogue DCOMP90 are exactly the on-map "
        "sources\", the same meaning Pokhrel et al. 2023, eHOPS I, ApJS 266, 32, "
        "gave the old catalogue column: a band with no completeness value of its "
        "own is a band not covered there, and the SED fitter's non-detection term "
        "excludes it)."),

    # population.young_stars: folded into population.anchor_weights.build
    # (brief U4) -- no product of its own, no registry entries. See
    # population.young_stars.build_region's own docstring for what
    # N_G_YOUNG/N_KS_YOUNG/N_YOUNG_TOTAL mean; figure_anchors recomputes
    # them in memory, population.anchor_observed.compute_region reads
    # them as the dict build_region returns, directly.

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
        "rule at this pixel's own depth, further split off STAR's own "
        "density (N_CAT_STAR + N_CAT_PAHC never exceeds the field-star "
        "density) by P_PAHC -- the fraction of catalogued stars at this "
        "brightness relative to the local 8 micron limit whose measured "
        "8 micron flux carries a nebular excess (population.pahc."
        "curve_pahc_survey P_Q_REGION, ledger C8)."),
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
    ("prior_atlas_hpx512", "N_CAT_PAHN"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the cataloged density of "
        "the PAHN (star-less PAH nebular source) population: its "
        "measured curve (population.pahn) at this pixel's own 8 micron "
        "50%-completeness limit, weighted by the probability of "
        "clearing SESNA's two-of-eight-band detection rule."),
    ("prior_atlas_hpx512", "INTENSITY_YSO"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, the YSO population's own "
        "dimmed intrinsic density at this pixel's own column, before the "
        "cataloged-detection probability N_CAT_YSO applies."),
    ("prior_atlas_hpx512", "N_ABOVE_STAR"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, how much of the STAR "
        "population's own intrinsic density lies above "
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
    ("prior_atlas_hpx512", "N_ABOVE_PAHN"): ("objects per square degree",
        "For the matching pixel in HPX_PIX_512, how much of the PAHN "
        "population's own intrinsic density lies above this pixel's own "
        "I2 (4.5 micron) 50%-completeness flux, read directly from the "
        "class's own stored shape rather than from the cataloged-"
        "detection fraction."),
    ("prior_atlas_hpx512", "N_CAT_CELL_GAL"): ("objects",
        "For the matching pixel in HPX_PIX_512, the background-galaxy "
        "population's own expected cataloged count on the common "
        "(log10 xi, log10 F_4.5) grid: summing every cell gives this "
        "pixel's own RATIO_GAL times the region's source count. The "
        "log10 F_4.5 position is each member's own DIMMED 4.5 micron "
        "flux at the pixel being counted (the same dimming its catalog "
        "probability already applies), not its dereddened flux: the "
        "cell a member's expected catalog count lands in is the cell "
        "the sky would actually catalog it in, so this array compares "
        "like with like against sky sources, themselves binned by "
        "measured flux (C29, briefs/round4_2026-10-20/"
        "C29_atlas_check_dimmed_axis.md). Read only by this check "
        "(atlas.shapes's own diagnostic page, and the external "
        "diagnostic script it matches); never by bmstp or fittp."),
    ("prior_atlas_hpx512", "N_CAT_CELL_STAR"): ("objects",
        "For the matching pixel in HPX_PIX_512, the STAR population's "
        "own expected cataloged count on the common (log10 xi, "
        "log10 F_4.5) grid: summing every cell gives this pixel's own "
        "RATIO_STAR times the region's source count. The log10 F_4.5 "
        "position is each member's own DIMMED 4.5 micron flux at the "
        "pixel being counted (the same dimming its catalog probability "
        "already applies), not its dereddened flux (C29, briefs/"
        "round4_2026-10-20/C29_atlas_check_dimmed_axis.md). Read only "
        "by this check (atlas.shapes's own diagnostic page, and the "
        "external diagnostic script it matches); never by bmstp or "
        "fittp."),
    ("prior_atlas_hpx512", "N_CAT_CELL_PAHC"): ("objects",
        "For the matching pixel in HPX_PIX_512, the PAHC population's "
        "own expected cataloged count on the common (log10 xi, "
        "log10 F_4.5) grid: summing every cell gives this pixel's own "
        "RATIO_PAHC times the region's source count. The log10 F_4.5 "
        "position is each member's own DIMMED 4.5 micron flux at the "
        "pixel being counted (the same dimming its catalog probability "
        "already applies), not its dereddened flux (C29, briefs/"
        "round4_2026-10-20/C29_atlas_check_dimmed_axis.md). Read only "
        "by this check (atlas.shapes's own diagnostic page, and the "
        "external diagnostic script it matches); never by bmstp or "
        "fittp."),
    ("prior_atlas_hpx512", "N_CAT_CELL_AGB"): ("objects",
        "For the matching pixel in HPX_PIX_512, the AGB population's own "
        "expected cataloged count on the common (log10 xi, log10 F_4.5) "
        "grid: summing every cell gives this pixel's own RATIO_AGB times "
        "the region's source count. The log10 F_4.5 position is each "
        "member's own DIMMED 4.5 micron flux at the pixel being counted "
        "(the same dimming its catalog probability already applies), "
        "not its dereddened flux (C29, briefs/round4_2026-10-20/"
        "C29_atlas_check_dimmed_axis.md). Read only by this check "
        "(atlas.shapes's own diagnostic page, and the external "
        "diagnostic script it matches); never by bmstp or fittp."),
    ("prior_atlas_hpx512", "N_CAT_CELL_YSO"): ("objects",
        "For the matching pixel in HPX_PIX_512, the YSO population's own "
        "expected cataloged count on the common (log10 xi, log10 F_4.5) "
        "grid: summing every cell gives this pixel's own RATIO_YSO times "
        "the region's source count. The log10 F_4.5 position is each "
        "member's own DIMMED 4.5 micron flux at the pixel being counted "
        "(the same dimming its catalog probability already applies), "
        "not its dereddened flux (C29, briefs/round4_2026-10-20/"
        "C29_atlas_check_dimmed_axis.md). Read only by this check "
        "(atlas.shapes's own diagnostic page, and the external "
        "diagnostic script it matches); never by bmstp or fittp."),
    ("prior_atlas_hpx512", "N_CAT_CELL_H2S"): ("objects",
        "For the matching pixel in HPX_PIX_512, the H2-shock "
        "population's own expected cataloged count on the common "
        "(log10 xi, log10 F_4.5) grid: summing every cell gives this "
        "pixel's own RATIO_H2S times the region's source count. The "
        "log10 F_4.5 position is each member's own DIMMED 4.5 micron "
        "flux at the pixel being counted (the same dimming its catalog "
        "probability already applies), not its dereddened flux (C29, "
        "briefs/round4_2026-10-20/C29_atlas_check_dimmed_axis.md). Read "
        "only by this check (atlas.shapes's own diagnostic page, and "
        "the external diagnostic script it matches); never by bmstp or "
        "fittp."),
    ("prior_atlas_hpx512", "N_CAT_CELL_PAHN"): ("objects",
        "For the matching pixel in HPX_PIX_512, the PAHN population's "
        "own expected cataloged count on the common (log10 xi, "
        "log10 F_4.5) grid: summing every cell gives this pixel's own "
        "RATIO_PAHN times the region's source count. The log10 F_4.5 "
        "position is each member's own DIMMED 4.5 micron flux at the "
        "pixel being counted (the same dimming its catalog probability "
        "already applies), not its dereddened flux (C29, briefs/"
        "round4_2026-10-20/C29_atlas_check_dimmed_axis.md). Read only "
        "by this check (atlas.shapes's own diagnostic page, and the "
        "external diagnostic script it matches); never by bmstp or "
        "fittp."),
    ("prior_atlas_hpx512", "N_CELL_STAR"): ("objects",
        "For the matching pixel in HPX_PIX_512, the STAR population's "
        "own unthinned intrinsic count on the common (log10 xi, "
        "log10 F_4.5) grid: the expected count with every member's own "
        "probability of being cataloged set to one (no flux cut, no "
        "dimming) -- the log10 F_4.5 position here is still each "
        "member's own UNDIMMED (dereddened) flux, unchanged by C29 "
        "(briefs/round4_2026-10-20/C29_atlas_check_dimmed_axis.md): "
        "since the two arrays' log10 F_4.5 axes now differ, N_CAT_CELL_"
        "STAR no longer stays beneath this cell by cell, only in total "
        "(N_CAT_CELL_STAR.sum() <= this array's own sum)."),
    ("prior_atlas_hpx512", "N_CELL_AGB"): ("objects",
        "For the matching pixel in HPX_PIX_512, the AGB population's own "
        "unthinned intrinsic count on the common (log10 xi, log10 F_4.5) "
        "grid: the expected count with every member's own probability "
        "of being cataloged set to one (no flux cut, no dimming) -- the "
        "log10 F_4.5 position here is still each member's own UNDIMMED "
        "(dereddened) flux, unchanged by C29 (briefs/round4_2026-10-20/"
        "C29_atlas_check_dimmed_axis.md): since the two arrays' log10 "
        "F_4.5 axes now differ, N_CAT_CELL_AGB no longer stays beneath "
        "this cell by cell, only in total (N_CAT_CELL_AGB.sum() <= "
        "this array's own sum)."),
    ("prior_atlas_hpx512", "N_CELL_PAHC"): ("objects",
        "For the matching pixel in HPX_PIX_512, the PAHC population's "
        "own unthinned intrinsic count on the common (log10 xi, "
        "log10 F_4.5) grid: the expected count with every member's own "
        "probability of being cataloged set to one (no flux cut, no "
        "dimming) -- the log10 F_4.5 position here is still each "
        "member's own UNDIMMED (dereddened) flux, unchanged by C29 "
        "(briefs/round4_2026-10-20/C29_atlas_check_dimmed_axis.md): "
        "since the two arrays' log10 F_4.5 axes now differ, N_CAT_CELL_"
        "PAHC no longer stays beneath this cell by cell, only in total "
        "(N_CAT_CELL_PAHC.sum() <= this array's own sum)."),
    ("prior_atlas_hpx512", "N_CELL_GAL"): ("objects",
        "For the matching pixel in HPX_PIX_512, the background-galaxy "
        "population's own unthinned intrinsic count on the common "
        "(log10 xi, log10 F_4.5) grid: the expected count with every "
        "member's own probability of being cataloged set to one (no "
        "flux cut, no dimming) -- the log10 F_4.5 position here is "
        "still each member's own UNDIMMED (dereddened) flux, unchanged "
        "by C29 (briefs/round4_2026-10-20/"
        "C29_atlas_check_dimmed_axis.md): since the two arrays' log10 "
        "F_4.5 axes now differ, N_CAT_CELL_GAL no longer stays beneath "
        "this cell by cell, only in total (N_CAT_CELL_GAL.sum() <= "
        "this array's own sum)."),
    ("prior_atlas_hpx512", "N_CELL_YSO"): ("objects",
        "For the matching pixel in HPX_PIX_512, the YSO population's own "
        "unthinned intrinsic count on the common (log10 xi, log10 F_4.5) "
        "grid: the expected count with every member's own probability "
        "of being cataloged set to one (no flux cut, no dimming) -- the "
        "log10 F_4.5 position here is still each member's own UNDIMMED "
        "(dereddened) flux, unchanged by C29 (briefs/round4_2026-10-20/"
        "C29_atlas_check_dimmed_axis.md): since the two arrays' log10 "
        "F_4.5 axes now differ, N_CAT_CELL_YSO no longer stays beneath "
        "this cell by cell, only in total (N_CAT_CELL_YSO.sum() <= "
        "this array's own sum)."),
    ("prior_atlas_hpx512", "N_CELL_H2S"): ("objects",
        "For the matching pixel in HPX_PIX_512, the H2-shock "
        "population's own unthinned intrinsic count on the common "
        "(log10 xi, log10 F_4.5) grid: the expected count with every "
        "member's own probability of being cataloged set to one (no "
        "flux cut, no dimming) -- the log10 F_4.5 position here is "
        "still each member's own UNDIMMED (dereddened) flux, unchanged "
        "by C29 (briefs/round4_2026-10-20/"
        "C29_atlas_check_dimmed_axis.md): since the two arrays' log10 "
        "F_4.5 axes now differ, N_CAT_CELL_H2S no longer stays beneath "
        "this cell by cell, only in total (N_CAT_CELL_H2S.sum() <= "
        "this array's own sum)."),
    ("prior_atlas_hpx512", "N_CELL_PAHN"): ("objects",
        "For the matching pixel in HPX_PIX_512, the PAHN population's "
        "own unthinned intrinsic count on the common (log10 xi, "
        "log10 F_4.5) grid: the expected count with every member's own "
        "probability of being cataloged set to one (no flux cut, no "
        "dimming) -- the log10 F_4.5 position here is still each "
        "member's own UNDIMMED (dereddened) flux, unchanged by C29 "
        "(briefs/round4_2026-10-20/C29_atlas_check_dimmed_axis.md): "
        "since the two arrays' log10 F_4.5 axes now differ, N_CAT_CELL_"
        "PAHN no longer stays beneath this cell by cell, only in total "
        "(N_CAT_CELL_PAHN.sum() <= this array's own sum)."),
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
        "(itself the field-star density split off STAR's by P_PAHC, the "
        "fraction of catalogued stars at this brightness relative to "
        "the local 8 micron limit whose measured 8 micron flux carries "
        "a nebular excess, population.pahc.curve_pahc_survey "
        "P_Q_REGION, ledger C8) divided by the sum of N_CAT_<class> "
        "over every class."),
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
    ("prior_atlas_hpx512", "SHARE_PAHN"): ("dimensionless",
        "For the matching pixel in HPX_PIX_512, the PAHN population's "
        "own share of this pixel's total cataloged density: N_CAT_PAHN "
        "divided by the sum of N_CAT_<class> over every class."),

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
    ("table_density_source", "D_PAHC"): ("dimensionless (log10 mJy)",
        "Minus log10 of this source's own 8 micron completeness-limit "
        "flux (`catalog.limits.limits`'s 8 micron column, the catalog's "
        "per-source limits product): the PAHC contamination test's own "
        "depth term, before dividing by a predicted photospheric flux."),
    ("table_density_source", "DENSITY_STAR"): ("objects per square degree",
        "This source's own retained STAR sky density: the field-star "
        "population's tile density at this source's own tile."),
    ("table_density_source", "DENSITY_AGB"): ("objects per square degree",
        "This source's own retained AGB sky density: the evolved-star "
        "population's tile density at this source's own tile."),
    ("table_density_source", "DENSITY_PAHC"): ("objects per square degree",
        "This source's own retained PAHC sky density, identical to "
        "DENSITY_STAR (PAHC shares the STAR population's own grid): "
        "the field-star density UNSPLIT by contamination. The fraction "
        "of catalogued stars at this brightness relative to the local 8 "
        "micron limit whose measured 8 micron flux carries a nebular "
        "excess (population.pahc.curve_pahc_survey P_Q_REGION, ledger "
        "C8) is not applied here -- it splits this same density into "
        "STAR-only and PAHC-only shares downstream, in bmstp.atlas's "
        "N_CAT_STAR/N_CAT_PAHC, never in this table."),
    ("table_density_source", "DENSITY_GAL"): ("objects per square degree",
        "This source's own retained background-galaxy sky density: the "
        "one survey-wide galaxy counts-law density, the same number for "
        "every source."),
    ("table_density_source", "DENSITY_YSO"): ("objects per square degree",
        "This source's own retained YSO sky density: the young-star law "
        "applied to this source's own cloud-foreground column (A_COL_K "
        "times the sightline's own foreground share of the column) and "
        "this source's own arm, times the sightline's own on-grid "
        "retained fraction."),
    ("table_density_source", "DENSITY_H2S"): ("objects per square degree",
        "This source's own retained H2-shock sky density: the intrinsic "
        "young-star law density (convolved with the region's own HGBS "
        "map for a Herschel-arm source the convolution reaches) times "
        "the region's own knot rate and universal external fraction, "
        "times the sightline's own on-grid retained fraction."),
    ("table_density_source", "DENSITY_PAHN"): ("objects per square degree",
        "This source's own retained PAHN (star-less PAH nebular source) "
        "sky density: population.pahn's own measured density curve, "
        "read at this source's own admitted pixel's 8 micron 50%-"
        "completeness limit, times the sightline's own on-grid retained "
        "fraction. No external calibration factor (unlike DENSITY_H2S's "
        "knot rate): PAHN's density is the measured curve and the "
        "shape's own retention alone. An upper bound: the measured "
        "curve's own pattern count also contains any true protostar too "
        "faint at 3.6 and 4.5 to detect, and artefacts."),

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
    ("yso_weights_region", "SUBCLASS_PROB_POP"): ("probability, each row summing to 1",
        "For every YSO template (row) and each of the six young-star "
        "evolutionary stages (column, same order as this survey's YSO "
        "stage list), the probability that a source truly at this "
        "template's spot in the model grid is that stage, once the "
        "model library has been reweighted to match this region's own "
        "measured mix of stages, including its diskless members (the "
        "'population' factor in this same file). A template that the "
        "model grid alone could not assign to a single stage keeps a "
        "split between stages here too, but rebalanced to the measured "
        "mix; a template the grid assigns to one stage outright keeps "
        "that single stage. This is the number the fit's own per-stage "
        "breakdown is built from for YSO sources; unlike the other "
        "datasets in this file it depends on the region, through its "
        "own disk fraction and age."),

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
        "For every AGB template and every cell of LOG10_F45_CENTERS, the "
        "chance that template is the one responsible for a star observed "
        "at that brightness: the region's own evolved-star population "
        "(luminosity and distance) pushed through every template's own "
        "flux-to-luminosity ratio and its own share of the Riebel et al. "
        "(2012) GRAMS optical-depth census (by dust chemistry), then "
        "normalised over template at each brightness cell -- a thick "
        "shell's templates dominate where that population lands bright, "
        "a thin shell's where it lands faint, rather than every template "
        "carrying the same share at every brightness. The SAME population "
        "and census also build population/star/star_shape_tile's "
        "GRID_AGB (summed over template there instead of kept): one "
        "measure, read two ways. The underlying census is LMC-derived, "
        "applied to this Galactic survey without recalibration, and "
        "carries no AGB time variability; neither is modelled. Which "
        "population statement this factor encodes is named by this "
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
        "type axis of its own, times the measured 8 micron excess-"
        "amplitude distribution of contaminated sources (population."
        "pahc.curve_pahc_survey P_Q_REGION x N_PER_BIN_REGION, ledger "
        "C8: the fraction of catalogued stars at this brightness "
        "relative to the local 8 micron limit whose measured 8 micron "
        "flux carries a nebular excess, times the bin's own population, "
        "i.e. the region's own expected contaminated-source count per "
        "bin -- not the measured-only excess count) read onto the "
        "template's own amplitude parameter, needing no further "
        "division by a library density. Which population statement "
        "this factor encodes is named by this dataset's own enclosing "
        "group's NAME attribute."),
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
        "this factor's own per-template weight: the template's share of "
        "the measured population of outflow knots (its pre-shock "
        "density and its infrared colour against catalogued knots), "
        "spread onto the brightness axis by the region's own knot-"
        "brightness distribution. Which population statement this "
        "factor encodes is named by this dataset's own enclosing "
        "group's NAME attribute."),
    ("h2shock_weights_region", "C_F"): ("dimensionless",
        "For every h2shock template, this factor's own per-template "
        "offset: a correction applied to the template's weight at read "
        "time, beyond the per-cell table in W."),

    ("pahn_weights_region", "MODEL_NAME"): ("source name",
        "This PAHN template's own identifier, the register's own model "
        "name, in this file's row order (the same register read every "
        "other class uses)."),
    ("pahn_weights_region", "C_THETA"): ("log10 mJy at 1 kpc",
        "This PAHN template's own offset onto the common brightness "
        "axis: the register's own F_REF_I2, floored at FLOOR_LINEAR, the "
        "SAME convention every other library but h2shock uses (survey-"
        "wide, not region-specific) -- the amplitude the fitter's own "
        "fitted log10 B is relative to, so log10 B_hat + C_THETA recovers "
        "the source's own I2 flux. The population-level brightness "
        "placement (the pattern sources' own measured flux) lives "
        "separately, in bmstp.shapes.build_cloud's GRID_PAHN and "
        "bmstp.atlas._pahn_region_nodes, independent of this table."),
    ("pahn_weights_region", "LOG10_F45_CENTERS"): ("log10 mJy at 1 kpc",
        "The centers of the 110 common brightness cells every factor "
        "table in this file is tabulated on."),
    ("pahn_weights_region", "W"): ("dimensionless",
        "For every PAHN template and every cell of LOG10_F45_CENTERS, "
        "this factor's own per-template weight: uniform, 1/n_model at "
        "every cell alike (one population, no axis the census can "
        "weight). Which population statement this factor encodes is "
        "named by this dataset's own enclosing group's NAME attribute."),
    ("pahn_weights_region", "C_F"): ("dimensionless",
        "For every PAHN template, this factor's own per-template "
        "offset: a correction applied to the template's weight at read "
        "time, beyond the per-cell table in W. Equal to C_THETA here "
        "(PAHN's single factor carries no second offset), the same "
        "convention h2shock's own single factor uses."),

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
        "out. Each star's own luminosity and distance is pushed through "
        "EVERY AGB shell template's own flux-to-luminosity ratio, "
        "weighted by that template's own share of its dust chemistry's "
        "population (an optical-depth census, Riebel et al. 2012), then "
        "summed over templates -- a thick shell lands bright and a thin "
        "shell faint, rather than every template sharing one pooled "
        "brightness. That census is LMC-derived, carried over to this "
        "Galactic survey without recalibration, and includes no AGB "
        "time variability; neither is modelled here."),
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
        "up to the floor ON_GRID_YSO leaves out. A two-component mixture "
        "(owner's ruling 2026-10-20): the excess-selected census's own "
        "brightness marginal (bmstp.sample_cloud.p_ref_f45) at weight "
        "DISK_FRACTION (this region's own row in population/yso/"
        "law_yso_region.hdf5), plus the diskless Class III members' "
        "photospheric brightness marginal (bmstp.sample_cloud."
        "photosphere_f45_marginal) at weight 1 - DISK_FRACTION; the "
        "depth placement, shift kernel and normalisation are identical "
        "for both components. Bit-identical to the single-component "
        "census grid where DISK_FRACTION = 1 (the identity case)."),
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
    ("cloud_shape_sightline", "GRID_PAHN"): ("dimensionless",
        "For the matching sightline in HPX_PIX_256, the PAHN (star-less "
        "PAH nebular source) population density on the common (log10 xi, "
        "log10 F_4.5) grid: separable, a region-wide brightness marginal "
        "from the pattern sources' own measured I4 flux (population.pahn), "
        "in this region's own flux coordinate, times THIS SIGHTLINE's own "
        "front-surface marginal (its own foreground share of the column "
        "at the cloud's front edge, u_front -- the same value a "
        "foreground star on this sightline sits at) -- integrating over a "
        "cell gives that cell's own share of the region's PAHN population, "
        "up to the floor ON_GRID_PAHN leaves out."),
    ("cloud_shape_sightline", "MASS_OUTSIDE_PAHN"): ("dimensionless",
        "For the matching sightline in HPX_PIX_256, the fraction of this "
        "sightline's own PAHN population weight that falls outside the "
        "common grid, not stored in GRID_PAHN -- varies by sightline "
        "(each sightline's own front-edge foreground column, u_front, "
        "differs)."),
    ("cloud_shape_sightline", "ON_GRID_PAHN"): ("dimensionless",
        "1 minus MASS_OUTSIDE_PAHN: the fraction of the region's own PAHN "
        "population weight that GRID_PAHN actually holds."),
    ("cloud_shape_sightline", "D_PEAK_SIGHTLINE_PC"): ("pc",
        "For the matching sightline in HPX_PIX_256, the distance of this "
        "sightline's own dust-structure peak in the 3-D extinction map "
        "(the local maximum nearest this region's adopted distance), used "
        "to anchor GRID_YSO's brightness shift so that a footprint with "
        "more than one structure along the line of sight (for instance "
        "Aquila's Serpens-Main and Aquila-Rift components) gets a "
        "different shift on each side. Equal to this region's own single "
        "D_PEAK_PC (sky/derived/edenhofer/depth/region) where no separable "
        "local peak is found."),
    ("cloud_shape_sightline", "XI_FRONT"): ("dimensionless",
        "For the matching sightline in HPX_PIX_256, its own cumulative "
        "extinction profile u = A(d)/A(inf) evaluated at this region's "
        "cloud interval's own front edge (this file's D_FRONT_PC "
        "attribute). Folded in from the retired bmstp.cloud_interval "
        "stage (brief U4)."),
    ("cloud_shape_sightline", "XI_BACK"): ("dimensionless",
        "For the matching sightline in HPX_PIX_256, its own cumulative "
        "extinction profile u = A(d)/A(inf) evaluated at this region's "
        "cloud interval's own back edge (this file's D_BACK_PC "
        "attribute). Never smaller than XI_FRONT. Folded in from the "
        "retired bmstp.cloud_interval stage (brief U4)."),
    ("cloud_shape_sightline", "W_CLOUD"): ("dimensionless",
        "For the matching sightline in HPX_PIX_256, the diffuse/dense "
        "extinction-law ramp weight evaluated at the column of dust "
        "between the cloud interval's front and back edges, (XI_BACK - "
        "XI_FRONT) times this sightline's own total extinction column. "
        "0 where that column is 0. Folded in from the retired "
        "bmstp.cloud_interval stage (brief U4)."),

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
    ("population_star_tile", "HPX_PIX_512"): ("nested HEALPix pixel, nside 512",
        "For every star this tile draws, the one anchor pixel of this "
        "tile it is placed on, drawn with probability proportional to "
        "the pixel's own solid angle (every nside-512 pixel shares one "
        "solid angle, so this is a uniform draw over the tile's own "
        "pixels). XI and W are both formed at this SAME pixel's own "
        "extinction column and sightline, not at a tile average: the "
        "weight fit and the weight assignment read one construction."),
    ("population_star_tile", "XI"): ("dimensionless, 0 to 1",
        "For every star this tile draws, its placement fraction along "
        "its own drawn pixel's (HPX_PIX_512) extinction profile: "
        "u = A(d)/(its own drawn pixel's column) at the star's own "
        "distance, on that pixel's own parent sightline, clipped at 1 "
        "(the far wall of the pencil beam) -- a DEPTH coordinate, read "
        "by bmstp as the star's own placement mark, never the "
        "extinction itself. A star beyond the Edenhofer map's own edge "
        "(C26 owner ruling 2026-10-21, briefs/round4_2026-10-20/"
        "C26_profile_adopted.md) can genuinely see more extinction than "
        "this pixel's own column implies -- that extra reddening is in "
        "A_STAR_K, this star's own stored extinction, not in XI, which "
        "this product always keeps at or below 1."),
    ("population_star_tile", "A_STAR_K"): ("mag A_K",
        "For every star this tile draws, its own local K-band extinction "
        "at its own distance, on its own drawn pixel's (HPX_PIX_512) "
        "extinction profile: the number this star's own predicted "
        "magnitudes, weight and PAHC contamination probability are all "
        "formed from (C26 owner ruling 2026-10-21). A reader never "
        "reconstructs this by multiplying a column from elsewhere by "
        "XI: beyond the Edenhofer map's own edge this star's own far-"
        "field reddening can exceed what any single pixel's own column "
        "times XI would give."),
    ("population_star_tile", "W"): ("dimensionless",
        "For every star this tile draws, its total anchor reweighting "
        "factor: the STAR anchor's own Gaia/2MASS joint or marginal "
        "weight at the star's own placement and predicted magnitude, "
        "both formed at its own drawn pixel's (HPX_PIX_512) extinction "
        "column and sightline -- the same construction the anchor "
        "weights were fitted on, not a tile-mean approximation. "
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
    ("population_star_tile", "RULE_COUNT"): ("star count",
        "C21 item 2: this tile's own star count under each of the six "
        "WEIGHT_RULE codes (module docstring: 0 joint, 1 G marginal "
        "only, 2 Ks marginal only, 3 faint end, 4 bright end, 5 both "
        "marginals, placement-selected) -- the same code every one of "
        "this tile's stars was assigned by star_weights, counted rather "
        "than stored star by star."),
    ("population_star_tile", "RULE_WEIGHT"): ("dimensionless",
        "C21 item 2: this tile's own sum of W under each of the six "
        "WEIGHT_RULE codes -- RULE_WEIGHT.sum() equals W.sum() for this "
        "tile exactly."),
    ("population_star_tile", "RULE_COUNT_FRAC"): ("dimensionless",
        "C21 item 2: this region's own star-count share under each of "
        "the six WEIGHT_RULE codes -- the sum, over every tile_<id> "
        "group's own RULE_COUNT, normalised to 1 -- so the share of the "
        "population the placement rule (code 5) decides is a plain "
        "number in this product, not something a reader has to sum "
        "RULE_COUNT themselves to see."),
    ("population_star_tile", "RULE_WEIGHT_FRAC"): ("dimensionless",
        "C21 item 2: this region's own WEIGHT share under each of the "
        "six WEIGHT_RULE codes -- the sum, over every tile_<id> group's "
        "own RULE_WEIGHT, normalised to 1."),
    ("population_star_tile", "POP"): ("category (0/1)",
        "C30 (briefs/round4_2026-10-20/C30_population_anchor.md) item 5: "
        "this star's own Galactic field-star population, population."
        "selection.is_giant's own HR-diagram cut on its TRILEGAL LOG_G -- "
        "0 dwarf, 1 giant. Independent of IS_EVOLVED (an AGB star can be "
        "a giant or not; a giant need not be evolved)."),
    ("population_star_tile", "RULE_COUNT_POP"): ("star count",
        "C30 item 5: this tile's own RULE_COUNT, split by POP -- shape "
        "(2, 6), row 0 dwarf stars, row 1 giant stars; RULE_COUNT_POP."
        "sum(axis=0) equals RULE_COUNT exactly."),
    ("population_star_tile", "RULE_WEIGHT_POP"): ("dimensionless",
        "C30 item 5: this tile's own RULE_WEIGHT, split by POP -- shape "
        "(2, 6); RULE_WEIGHT_POP.sum(axis=0) equals RULE_WEIGHT exactly."),
    ("population_star_tile", "RULE_COUNT_FRAC_POP"): ("dimensionless",
        "C30 item 5: this region's own RULE_COUNT_POP, summed over every "
        "tile_<id> group and normalised WITHIN each population row (each "
        "row sums to 1 over that population's own stars, not over the "
        "region total) -- so a giant's own rule-code share and a dwarf's "
        "are each read on their own population's own 100%."),
    ("population_star_tile", "RULE_WEIGHT_FRAC_POP"): ("dimensionless",
        "C30 item 5: the same construction as RULE_COUNT_FRAC_POP, on "
        "RULE_WEIGHT_POP (each row normalised within its own population's "
        "own total weight) instead of the star count."),
    # C21's own KS_EDGES_2MASS/N_KS_OBS_2MASS/FINAL_POP_KS_HIST build-time
    # check (commit 701fe92) is removed (C23 brief item 4, owner ruling):
    # it dimmed every star at its tile's own MEAN column and summed W
    # without solid-angle weighting, so it read 1.0 where the atlas-
    # grounded check (bmstp.atlas, real per-pixel columns, solid-angle
    # weighted) read 0.63 -- a check that cannot see the defect it was
    # meant to catch is worse than none.

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
        "tabulated on. For a region with no deep (UKIDSS GPS or VVV) "
        "coverage of its own, the faintest bins past its own 2MASS "
        "depth are a disclosed extrapolation (see W_REGION_KS), not an "
        "observed anchor."),
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
        "value, or NaN where neither this region nor the survey has any. "
        "A region with no deep coverage of its own carries no real Ks "
        "anchor past 2MASS's own depth; its faintest bins are instead "
        "its own last measured value stepped by the faint-end slope "
        "measured on the OTHER, deep-covered regions (one survey-wide "
        "number, FAINT_TREND_KS_DEX_PER_MAG with FAINT_TREND_KS_"
        "TRANSPORTED True) -- a disclosed extrapolation, not a fit."),
    ("weights_anchors_tile", "POPULATED_KS"): ("boolean",
        "One value per Ks bin of KS_EDGES: True where at least one tile "
        "of this region carries usable evidence in that bin, so "
        "W_REGION_KS is this region's own fit rather than a survey-pooled, "
        "transported-slope, or absent value."),
    ("weights_anchors_tile", "MEASURED_KS"): ("boolean",
        "For each tile and each Ks bin of KS_EDGES, True where the "
        "model predicts at least one star there: False marks a bin past "
        "this tile's own survey depth (a deep UKIDSS/VVV bin the region "
        "carries no deep coverage for, or a faint bin this file added "
        "beyond the region's own anchor depth by extrapolation), where a "
        "consumer reads W_KS/W_REGION_KS as a disclosed extrapolation, "
        "never as an observed anchor."),
    ("weights_anchors_tile", "W_JOINT"): ("dimensionless",
        "For each tile, each Gaia G bin of G_EDGES and each Ks bin of "
        "KS_EDGES, the anchor reweighting factor on the joint (G, Ks) "
        "grid: the sky's own observed star count over the model's "
        "predicted count in that joint cell, shrunk toward the region-"
        "pooled value in the log, or, where no tile of this region "
        "cleared the fitting floor on its own but this region's tiles "
        "pooled did (JOINT_SOURCE 2), the one region-pooled value every "
        "tile shares there. Read only where USE_JOINT is True for that "
        "tile and bin; a star without a usable joint cell takes the "
        "marginal weight instead."),
    ("weights_anchors_tile", "USE_JOINT"): ("boolean",
        "For each tile, each Gaia G bin of G_EDGES and each Ks bin of "
        "KS_EDGES, True where this joint cell carries evidence at "
        "either level (JOINT_SOURCE 1 or 2) and W_JOINT is therefore an "
        "actual fit rather than an unfitted placeholder -- at level 1 "
        "(a tile of this region cleared the floor on its own) also "
        "requiring this tile's own observed joint count to be positive; "
        "at level 2 (no tile did, but this region's tiles pooled did) "
        "every tile of the region reads it, since a tile's own zero "
        "crossmatch stars there is not evidence against a value the "
        "region's own tiles, summed, did measure (C21 item 1)."),
    ("weights_anchors_tile", "W_REGION_JOINT"): ("dimensionless",
        "One value per joint (G, Ks) cell (shared by every tile): the "
        "region-pooled joint reweighting factor, or, where no tile in "
        "this region has usable evidence in that cell, the survey-"
        "pooled value, or NaN where neither this region nor the survey "
        "has any."),
    ("weights_anchors_tile", "JOINT_SOURCE"): ("category (0/1/2)",
        "One value per joint (G, Ks) cell (shared by every tile; C21 "
        "item 1): 0 no joint evidence at any level (a star landing here "
        "falls to the marginal/placement rule, population.star_"
        "population.star_weights' WEIGHT_RULE 5); 1 at least one tile "
        "of this region cleared MIN_COUNTS on its own; 2 no single tile "
        "did, but this region's own tiles, summed (region_pooled_"
        "ratio), did."),
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
    ("weights_anchors_tile", "W_GIANT"): ("dimensionless",
        "C30 (briefs/round4_2026-10-20/C30_population_anchor.md): one value "
        "for this whole region -- TRILEGAL's model error against the "
        "independent anchors is per POPULATION, not per Ks magnitude alone "
        "(distant red giants over-produced, dwarfs close to right), so the "
        "2MASS colour cells at Ks 12.0-14.3 are fit jointly for a giant "
        "weight (LOG_G < 3.5, population.selection.is_giant) and this "
        "file's own per-tile, per-bin W_KS/W_REGION_KS there read as the "
        "dwarf weight. NaN where SPLIT is False: the region's own giant "
        "model count never cleared N_GIANT_MIN, and a giant star reads its "
        "bin's W_KS/W_REGION_KS exactly like a dwarf (population."
        "star_population)."),
    ("weights_anchors_tile", "W_GIANT_ERR"): ("dimensionless",
        "The joint weighted-least-squares fit's own formal error on "
        "W_GIANT (sigma^2 = max(observed colour-cell count, 1), no "
        "completeness factor). NaN where SPLIT is False."),
    ("weights_anchors_tile", "N_GIANT_MODEL"): ("stars",
        "The region's own model giant count summed over the 2MASS colour "
        "cells (Ks 12.0-14.3, every J-Ks bin), non-excluded tiles only -- "
        "the number SPLIT's own N_GIANT_MIN floor is measured against, "
        "printed in this module's own report regardless of whether the "
        "floor was cleared."),
    ("weights_anchors_tile", "SPLIT"): ("boolean",
        "True where this region's own N_GIANT_MODEL cleared N_GIANT_MIN "
        "and W_GIANT/W_GIANT_ERR are a real fit; False where there is no "
        "split at all (W_GIANT/W_GIANT_ERR are NaN and every star, giant "
        "or dwarf, reads this file's own per-bin W_KS/W_REGION_KS exactly "
        "as the pre-C30 construction did)."),

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
    ("histograms_anchors_hpx512", "KS_EDGES"): ("mag 2MASS/UKIDSS/VHS Ks",
        "The edges of the Ks magnitude bins N_KS_OBS, N_KS_PRED and "
        "N_GK_PRED are tabulated on, on the shared 2MASS/UKIDSS/VHS "
        "scale KS_SOURCE names per bin."),
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
        "KS_EDGES, the observed 2MASS/UKIDSS/VHS anchor star count, from "
        "the external anchor-count products KS_SOURCE names per bin "
        "(UKIDSS and VHS counts already multiplied by their own "
        "CALIBRATION[0] against 2MASS, C19)."),
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
        "For the matching pixel in HPX_PIX_512, True where EITHER deeper "
        "survey -- UKIDSS GPS or VHS (C19 Part B) -- reaches that pixel, "
        "so KS_EDGES runs past the 2MASS-only cut and KS_SOURCE marks "
        "the deep bins for it."),
    ("histograms_anchors_hpx512", "KS_SOURCE"): ("code",
        "For each Ks bin of KS_EDGES (one value per bin, shared by every "
        "pixel), which survey that bin's counts actually came from: 0 "
        "2MASS, 1 UKIDSS, 2 VHS (C19 Part B) -- UKIDSS is taken on a "
        "pixel it covers, VHS only where UKIDSS does not; the code names "
        "whichever side contributed the larger measured count to that "
        "bin. A region with no deep coverage at all has every bin 0."),
    ("histograms_anchors_hpx512", "RED_TAIL_SHARE_OBS"): ("dimensionless",
        "C25 item 5 (briefs/round4_2026-10-20/C25_far_column.md), a check, "
        "not a model input: the region-wide share of SESNA sources in "
        "this product's own anchor pixels with a measured 2MASS Ks in "
        "12-14 and a measured 2MASS J whose J-Ks exceeds 1.5 -- the "
        "missing 'red tail' of distant, deeply reddened background stars "
        "C24 located (briefs/round4_2026-10-20/C24_locate_red_tail.md). "
        "Written once population.star_population has built this region "
        "(a later stage than this product's own build)."),
    ("histograms_anchors_hpx512", "RED_TAIL_SHARE_MODEL"): ("dimensionless",
        "C25 item 5, the same check's model side: the built population's "
        "own area-weighted share of Ks 12-14 stars with J-Ks > 1.5, each "
        "star re-dimmed at its own drawn anchor pixel with sky.derived."
        "profile.a_of_d, uncapped at that pixel's own column -- the "
        "Marshall-informed far-field column (C25) on a covered sightline, "
        "unchanged on one Marshall does not reach. Compare to "
        "RED_TAIL_SHARE_OBS; an uncovered region's own construction is "
        "exactly as it was before C25."),
    ("histograms_anchors_hpx512", "CELL_KS_EDGES"): ("mag 2MASS Ks",
        "C30 (briefs/round4_2026-10-20/C30_population_anchor.md): the "
        "five half-mag Ks bins N_GIANT_CELL/N_DWARF_CELL are tabulated "
        "on, 12.0 to 14.3 -- the SAME grid sky.derived.twomass_counts' "
        "own CELL_KS_EDGES carries, read from there so the two never "
        "drift apart."),
    ("histograms_anchors_hpx512", "CELL_JK_EDGES"): ("mag J-Ks (2MASS)",
        "C30: the three interior J-Ks colour edges N_GIANT_CELL/"
        "N_DWARF_CELL's own last axis is tabulated on -- the SAME values "
        "sky.derived.twomass_counts' own CELL_JK_EDGES carries."),
    ("histograms_anchors_hpx512", "N_GIANT_CELL"): ("stars",
        "C30 item 2: for the matching pixel in HPX_PIX_512, each Ks bin "
        "of CELL_KS_EDGES and each J-Ks colour bin of CELL_JK_EDGES, the "
        "model's predicted GIANT (LOG_G < 3.5, population.selection."
        "is_giant) star count in that cell -- the retained field-star "
        "table's own population at this pixel's own column (NOT the RAW "
        "group N_KS_PRED reads: that group carries no LOG_G or flux, "
        "module docstring above '_acceptance_check_one_pixel'), nearest-"
        "pointing placed and weighted exactly as N_KS_PRED, but over a "
        "detection-filtered (retained) sample -- its own sum over both "
        "populations and all four colour bins is NOT bit-comparable to "
        "N_KS_PRED at the same Ks bin. Read by population.anchor_weights "
        "to fit a separate W_GIANT."),
    ("histograms_anchors_hpx512", "N_DWARF_CELL"): ("stars",
        "C30 item 2: the same cell and construction as N_GIANT_CELL, the "
        "model's predicted DWARF (LOG_G >= 3.5) star count instead."),
    ("histograms_anchors_hpx512", "N_GIANT_KS"): ("stars",
        "C30 item 3: one value per Ks bin of KS_EDGES (region-summed, "
        "not per pixel): the model's predicted GIANT star count on the "
        "SAME full Ks grid N_KS_PRED is tabulated on (not just the "
        "colour-cell range), from the SAME retained-table population as "
        "N_GIANT_CELL -- read only to disclose the deep bins' (Ks >= "
        "KS_SHARED_END_MAG) giant share, the assumption the deep bins "
        "stay dwarf-only by the model's own content rests on."),
    ("histograms_anchors_hpx512", "N_DWARF_KS"): ("stars",
        "C30 item 3: the same construction as N_GIANT_KS, the DWARF "
        "star count instead."),

    # population.gal -- counts_gal_survey: the background-galaxy number-
    # counts law, survey-wide.
    ("counts_gal_survey", "LOG10_A"): ("log10(deg^-2 mJy^-1)",
        "The fitted broken power law's own normalization: log10 of the "
        "DIFFERENTIAL galaxy count per square degree per mJy, phi(S), at "
        "the break flux LOG10_S_BREAK. Fitted directly to Fazio et al. "
        "(2004)'s own tabulated differential counts, never through an "
        "intermediate cumulative fit."),
    ("counts_gal_survey", "LOG10_S_BREAK"): ("log10 mJy",
        "The fitted broken power law's own break flux at 4.5 micron: "
        "10**x is a flux in mJy. Free to land anywhere the differential "
        "counts data support, not constrained to Fazio's own tabulated "
        "flux range."),
    ("counts_gal_survey", "ALPHA_FAINT"): ("dimensionless",
        "The fitted broken power law's own faint-end slope of log10 "
        "phi(S) (the differential counts) against log10 S."),
    ("counts_gal_survey", "ALPHA_BRIGHT"): ("dimensionless",
        "The fitted broken power law's own bright-end slope of log10 "
        "phi(S) (the differential counts) against log10 S."),
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
        "micron. Fitted directly to Fazio's own tabulated differential "
        "counts (never a cumulative fit then differentiated); its level "
        "agrees with a direct integral of that same table to within 0.01 "
        "dex over this grid."),
    ("counts_gal_survey", "P_POINT"): ("dimensionless",
        "The fraction of galaxies at each flux in LOG10_S_GRID retained "
        "as IRAC point sources rather than resolved and excluded. "
        "MEASURED directly on SWIRE's own adopted galaxies' per-band "
        "extended-source flag over 0.3-20 mJy (POINT_FRACTION_LOG10_S_"
        "CENTERS/MEASURED/N_SWIRE below carry that measurement); held at "
        "1 below 0.3 mJy, where the galaxy angular-size distribution has "
        "already crossed the IRAC point-spread function."),
    ("counts_gal_survey", "PHI_S_POINT"): ("deg^-2 mJy^-1",
        "PHI_S times P_POINT: the point-source-corrected galaxy counts "
        "law every consumer of this product reads."),
    ("counts_gal_survey", "COSMIC_VARIANCE_DEX"): ("dex",
        "The root-mean-square, in dex, of the counts-law fit's own "
        "field-to-field scatter over its candidate fitting variants: the "
        "cosmic-variance uncertainty on PHI_S and PHI_S_POINT."),
    ("counts_gal_survey", "FIT_LOG10_S_MIN"): ("log10 mJy",
        "The faintest flux Fazio et al. (2004)'s own tabulated bins "
        "reach: 10**x is a flux in mJy. Descriptive only -- the fitted "
        "break is free and not constrained to this range."),
    ("counts_gal_survey", "FIT_LOG10_S_MAX"): ("log10 mJy",
        "The brightest flux Fazio et al. (2004)'s own tabulated bins "
        "reach: 10**x is a flux in mJy. Descriptive only -- the fitted "
        "break is free and not constrained to this range."),
    ("counts_gal_survey", "FIT_RMS_DEX"): ("dex",
        "The root-mean-square residual, in dex, of the fitted broken "
        "power law against Fazio et al. (2004)'s own tabulated "
        "DIFFERENTIAL counts (the quantity actually fitted)."),
    ("counts_gal_survey", "POINT_FRACTION_LOG10_S_CENTERS"): ("log10 mJy",
        "The 4.5 micron flux bin centers the measured point-like "
        "fraction (POINT_FRACTION_MEASURED) is tabulated on, over SWIRE's "
        "adopted galaxies between 0.3 and 20 mJy: 10**x is a flux in "
        "mJy. P_POINT is this curve interpolated onto LOG10_S_GRID."),
    ("counts_gal_survey", "POINT_FRACTION_MEASURED"): ("dimensionless",
        "The measured fraction of SWIRE's adopted galaxies in each "
        "POINT_FRACTION_LOG10_S_CENTERS bin whose own per-band "
        "extended-source flag (3.6, 4.5 micron) never reads resolved -- "
        "the point-like fraction actually measured, never extrapolated, "
        "beneath P_POINT."),
    ("counts_gal_survey", "POINT_FRACTION_N_SWIRE"): ("galaxies",
        "How many SWIRE adopted galaxies fall in each "
        "POINT_FRACTION_LOG10_S_CENTERS bin: the sample size behind each "
        "POINT_FRACTION_MEASURED value."),

    # population.anchor_observed: folded into population.anchor_weights.
    # build (brief U4) -- no product of its own, no registry entries.
    # N_G_SUB/N_KS_SUB/N_GK_SUB (the model's own expected young-star and
    # cluster-member count subtracted from the observed Gaia/2MASS
    # histogram, floored at zero) live only in the dict
    # `anchor_observed.compute_region` returns, read directly by
    # `anchor_weights.region_tile_counts`.

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
        "How many excess-selected census objects (sesnaimpute.sky.derived."
        "yso_census, every ingested census of that kind, not Dunham et al. "
        "2015 alone -- owner's 2026-10-20 ruling), of any class, fall "
        "inside this region's own fit footprint: the union of the "
        "overlapping census(es)' own coverage polygon(s), each restricted "
        "to its own SURVEY_FLOOR_A_K where that is finite. A protostar-"
        "only census (Kryukova2014/Cygnus X) is never counted here -- see "
        "N_CENSUS_PROTOSTAR_ONLY."),
    ("law_yso_region", "FIT_A_K_MIN"): ("mag A_K",
        "A report value, not itself part of the fit footprint test any "
        "more (owner's 2026-10-20 ruling: the footprint is the census "
        "coverage polygon union, each restricted to its own "
        "SURVEY_FLOOR_A_K): the gas-column floor of whichever (SOURCE, "
        "CLOUD) census group this region's admitted footprint holds the "
        "most objects from (0.22 mag for a Dunham et al. 2015 c2d cloud, "
        "0.33 mag for a Gould Belt cloud, NaN for every other census "
        "here, which states no comparable coverage rule). NaN where no "
        "excess census group overlaps this region at all."),
    ("law_yso_region", "CENSUS_SOURCES"): ("';'-joined census names",
        "Which published census(es) (sesnaimpute.sky.derived.yso_census's "
        "own SOURCE values) overlap this region's admitted footprint and "
        "feed N_CENSUS_REGION, joined by ';' (e.g. 'Dunham2015', or "
        "'Megeath2012' for Orion A/B); empty where none overlap."),
    ("law_yso_region", "N_CENSUS_PROTOSTAR_ONLY"): ("protostars",
        "How many protostar-only census objects (sesnaimpute.sky.derived."
        "yso_census's own SELECTION == 'protostar' rows, today only "
        "Kryukova2014/Cygnus X) fall inside this region's own fit "
        "footprint -- reported separately from N_CENSUS_REGION; the "
        "owner's 2026-10-20 ruling leaves how a protostar-only census "
        "enters a region's fit to the law unit, not decided by this "
        "column's own existence. Zero where none fall there or no "
        "protostar-only census overlaps the region at all."),
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
        "(protostellar), independent of region: a share of the EXCESS-"
        "SELECTED census alone, unaffected by DISK_FRACTION (see "
        "CLASS_SHARE_WEAK's own reading for how this and the diskless "
        "share connect)."),
    ("law_yso_region", "CLASS_SHARE_DISK"): ("dimensionless",
        "The fraction of the whole Dunham et al. (2015) census with "
        "-1.6 <= alpha0 < -0.3 (disk-bearing), independent of region: a "
        "share of the EXCESS-SELECTED census alone, unaffected by "
        "DISK_FRACTION (see CLASS_SHARE_WEAK's own reading)."),
    ("law_yso_region", "CLASS_SHARE_WEAK"): ("dimensionless",
        "The fraction of the whole Dunham et al. (2015) census with "
        "alpha0 < -1.6 (weak or no disk), independent of region. This "
        "and CLASS_SHARE_PROTO/DISK are shares of the EXCESS-SELECTED "
        "census alone (the three sum to 1 by themselves); the fourth, "
        "diskless share, CLASS_SHARE_DISKLESS, is not part of this "
        "census-internal split -- the two bookkeeping schemes are "
        "connected only through DISK_FRACTION, below."),
    ("law_yso_region", "DISK_FRACTION"): ("dimensionless",
        "This region's own disk fraction (owner's ruling 2026-10-20): "
        "the fraction of the region's young stars that still carry an "
        "infrared excess at the region's own published age -- MEASURED "
        "from an excess-independent membership census where one covers "
        "the field, else the Richert et al. 2018 relation at the "
        "region's age (regions.Region.disk_fraction, .age_myr, "
        ".disk_fraction_source cites which and the paper). WHICH YOUNG "
        "STARS this fraction counts (whether protostars sit in its own "
        "numerator and denominator, or are excluded from both) is "
        "regions.Region.disk_fraction_definition ('all' or 'disks'), "
        "read by N_LAW_TOTAL_FACTOR's own formula, not by this dataset. "
        "1.0 (the identity case) for a region with no published age or "
        "disk fraction applied yet: every number downstream (N_LAW_"
        "TOTAL_FACTOR, CLASS_SHARE_DISKLESS, and every reader of the "
        "young-star law's level) is then unchanged from this ruling's "
        "own branch."),
    ("law_yso_region", "N_LAW_TOTAL_FACTOR"): ("dimensionless",
        "The factor population.yso.law_count and population.yso."
        "law_area_integral multiply the excess-selected young-star law "
        "by to recover the TOTAL young-star count, excess-selected "
        "members plus the diskless Class III members the excess count "
        "implies at this region's own disk fraction (never applied by "
        "population.knot_rate.region_predicted_yso, which stays excess-"
        "only by design). The formula depends on this region's own "
        "regions.Region.disk_fraction_definition: '1 / DISK_FRACTION' "
        "where the published fraction counts protostars in both its "
        "own numerator and denominator ('all'); 'CLASS_SHARE_PROTO + "
        "(1 - CLASS_SHARE_PROTO) / DISK_FRACTION' where protostars are "
        "excluded from both ('disks') -- the excess law's own "
        "protostar share then stays a share of the excess-selected "
        "count, and only the remaining disk-or-diskless share is "
        "inflated by 1 / DISK_FRACTION. Both give 1.0 (no change) where "
        "DISK_FRACTION is the identity case."),
    ("law_yso_region", "CLASS_SHARE_DISKLESS"): ("dimensionless",
        "1 - 1/N_LAW_TOTAL_FACTOR: this region's own diskless Class III "
        "members as a share of the TOTAL young-star population (not of "
        "the excess-selected census: CLASS_SHARE_PROTO/DISK/WEAK, each "
        "divided by N_LAW_TOTAL_FACTOR, together with this share sum to "
        "1 over the total population, in both the 'all' and 'disks' "
        "disk_fraction_definition cases alike). 0.0 where DISK_FRACTION "
        "is the identity case -- no diskless members are added. Feeds "
        "bmstp.template_weights' own fourth {CIII} census group, "
        "region by region, on top of (not in place of) the survey-wide "
        "CLASS_SHARE_WEAK group, and bmstp.atlas's own YSO member draw "
        "(_yso_region_weight)."),

    # population.pahc_curve -- curve_pahc_survey: the PAH-contamination
    # probability curve P(q), one row per region (WP-POP-4, ledger C7:
    # the excess rate is regional by a factor of 8, so one survey-wide
    # curve is wrong).
    ("curve_pahc_survey", "LOG10_Q_EDGES"): ("log10(dimensionless)",
        "The edges of the 40 bins in log10(q) this curve is tabulated on, "
        "q being a source's own 8 micron completeness limit divided by "
        "its predicted photospheric 8 micron flux: bin j runs from "
        "LOG10_Q_EDGES[j] to LOG10_Q_EDGES[j+1]."),
    ("curve_pahc_survey", "REGION"): ("name",
        "The region name of each row of every per-region array in this "
        "product, in the order those arrays are indexed on their first "
        "axis."),
    ("curve_pahc_survey", "P_Q_REGION"): ("dimensionless",
        "For each region (rows, REGION) and bin of LOG10_Q_EDGES "
        "(columns): the fraction of catalogued stars at this brightness "
        "relative to the local 8 micron limit whose measured 8 micron "
        "flux carries a nebular excess -- the probability every "
        "consumer applies to EVERY source at that q, not just the ones "
        "with an 8 micron measurement. Built from that region's own "
        "measured-only excess rate among its 8-micron-measured sources "
        "at that q, with the survey's quiescent-region noise floor "
        "(F_FLOOR) subtracted and floored at zero, THEN multiplied by "
        "that bin's own measured share M_PER_BIN_REGION/N_PER_BIN_REGION "
        "(ledger C8, 2026-10-20: a source with no 8 micron measurement "
        "is not assumed uncontaminated, so the measured-only rate is "
        "charged only against the measured share of the bin, not every "
        "source in it). Zero for every bin whose q is too small for a "
        "3-sigma nebular excess to register at all (see "
        "P_Q_REGION_BRIGHT_EXCESS). The measured-only rate itself, "
        "before this scaling, is P_Q_REGION_MEASURED_ONLY."),
    ("curve_pahc_survey", "P_Q_REGION_RAW"): ("dimensionless",
        "The same construction as P_Q_REGION before the quiescent-region "
        "noise floor is subtracted: that region's own measured-only "
        "excess rate at that q, multiplied by that bin's own measured "
        "share M_PER_BIN_REGION/N_PER_BIN_REGION -- the unfloored excess "
        "count over ALL of the bin's sources, not the measured-only rate "
        "(that is P_Q_REGION_MEASURED_ONLY)."),
    ("curve_pahc_survey", "P_Q_REGION_MEASURED_ONLY"): ("dimensionless",
        "For each region and bin of LOG10_Q_EDGES, that region's own "
        "excess fraction among ONLY its 8-micron-measured sources at "
        "that q (P_Q_REGION's own numerator statistic before it is "
        "scaled by the bin's measured share and floor-subtracted): kept "
        "for a reader checking the measured subset specifically. This is "
        "NOT the probability applied to a source -- a source with no 8 "
        "micron measurement is not assumed to carry this rate, or any "
        "assumed-zero rate either; P_Q_REGION is what every consumer "
        "reads."),
    ("curve_pahc_survey", "N_PER_BIN_REGION"): ("sources",
        "For each region and bin of LOG10_Q_EDGES, how many of that "
        "region's own sources with measured 3.6 and 4.5 micron "
        "photometry and no 4.5 micron excess fall in it: that region's "
        "own eligible population at that q."),
    ("curve_pahc_survey", "M_PER_BIN_REGION"): ("sources",
        "For each region and bin of LOG10_Q_EDGES, how many of "
        "N_PER_BIN_REGION's sources also have a measured 8 micron flux: "
        "the subset P_Q_REGION's excess fraction is actually measured "
        "on."),
    ("curve_pahc_survey", "P_Q_REGION_BRIGHT_EXCESS"): ("dimensionless",
        "For each region and bin of LOG10_Q_EDGES whose q is too small "
        "for a 3-sigma nebular excess to register (where P_Q_REGION is "
        "zeroed for that reason), that region's own unfloored excess "
        "fraction over ALL of the bin's sources (P_Q_REGION_RAW's own "
        "construction, measured-only rate times the bin's measured "
        "share): not contamination but circumstellar 8 micron emission, "
        "a source with no 8 micron measurement no more presumed free of "
        "it than of contamination elsewhere on the curve. Zero for "
        "every other bin."),
    ("curve_pahc_survey", "N_PER_BIN_REGION_BRIGHT_EXCESS"): ("sources",
        "For each region and bin of LOG10_Q_EDGES whose q is too small "
        "for a 3-sigma nebular excess to register, that region's own "
        "eligible population there. Zero for every other bin."),
    ("curve_pahc_survey", "P_Q_DISK_EXCESS"): ("dimensionless",
        "For each bin of LOG10_Q_EDGES, pooled over every region, the "
        "measured-only excess fraction among the 8-micron-measured "
        "sources that DO carry a 4.5 micron excess over their "
        "photospheric prediction (disk-bearing or otherwise "
        "circumstellar): reported beside the shipped, no-4.5-micron- "
        "excess curve, not subject to its noise floor."),
    ("curve_pahc_survey", "N_PER_BIN_DISK_EXCESS"): ("sources",
        "For each bin of LOG10_Q_EDGES, how many sources with a 4.5 "
        "micron excess and measured 3.6 and 4.5 micron photometry fall "
        "in it: P_Q_DISK_EXCESS's own denominator."),

    # population.pahn -- density_pahn_region: the star-less PAH nebular
    # source population's own sky density DENSITY_PAHN(limit), one row
    # per region, bins on one shared axis (C14_pahn_population.md).
    ("density_pahn_region", "LOG10_LIM_EDGES"): ("log10 mJy",
        "The edges of the RAW, fixed-width bins every region's own "
        "per-pixel counting starts from, in log10(the pixel's own 8 "
        "micron 50%-completeness flux, mJy): bin j runs from "
        "LOG10_LIM_EDGES[j] to LOG10_LIM_EDGES[j+1], 0.1 dex wide, one "
        "shared axis for every region in this file -- kept for "
        "provenance; no per-bin dataset below is tabulated on this axis "
        "any more (MERGED_LOG10_LIM_EDGES is, DEFECT 2, bms_review/"
        "briefs/round6_2026-10-22/U13_limit_coverage.md)."),
    ("density_pahn_region", "REGION"): ("name",
        "The region name of each row of every per-region array in this "
        "product, in the order those arrays are indexed on their first "
        "axis."),
    ("density_pahn_region", "MERGED_LOG10_LIM_EDGES"): ("log10 mJy",
        "For each region (REGION), its OWN merged-bin edges -- adjacent "
        "LOG10_LIM_EDGES bins merged forward until each holds at least "
        "PAHN_MIN_PER_BIN (10) raw pattern sources (population.pahn."
        "merge_region_bins, DEFECT 2: a raw 0.1-dex bin with too few "
        "sources is small-number noise, not a real dip in DENSITY_PAHN). "
        "A RAGGED dataset: one variable-length row per region, since one "
        "region's own merged-bin count differs from another's -- row i "
        "has length (number of that region's own merged bins) + 1, and "
        "is the axis every OTHER per-bin dataset below (DENSITY_PAHN, "
        "DENSITY_PAHN_CATALOGUED, N_PER_BIN, POISSON_LO, POISSON_HI, "
        "AREA_DEG2_PER_BIN) shares its own row i's length with."),
    ("density_pahn_region", "DENSITY_PAHN"): ("objects per square degree",
        "For each region (rows, REGION), one value per its own MERGED "
        "bin (MERGED_LOG10_LIM_EDGES's own row, a ragged dataset): the "
        "INTRINSIC sky density of the PAHN pattern -- a catalogued "
        "source with a measured 5.8 and 8 micron flux and no measured "
        "3.6 or 4.5 micron flux -- among this region's own admitted "
        "pixels whose own 8 micron 50%-completeness limit falls in this "
        "merged bin: each such pattern source's own weight, the "
        "inverse of its own probability of the catalogued event -- its "
        "own 5.8 and 8 micron flux both detected (weight_pahn_source's "
        "own P_CAT/WEIGHT, the same survey detection-probability mixture "
        "bmstp.atlas uses) -- summed there and divided by the total "
        "covered area there: the catalogued pattern count (DENSITY_PAHN_"
        "CATALOGUED) divided once by selection INTO THE PATTERN ITSELF, "
        "so a consumer that separately applies that same probability (as "
        "bmstp.atlas/the fitter do for a drawn member) applies it exactly "
        "once in total, not twice -- intrinsic for the PATTERN population "
        "only. A merged bin still at exactly zero (every one of its own "
        "raw bins empty, even after merging) is filled by log-linear "
        "interpolation if interior, held as measured at either end "
        "(DEFECT 2's own residual case). Still an upper bound, in both "
        "directions: the pattern count also contains any true protostar "
        "too faint at 3.6 and 4.5 to detect, and artefacts -- this "
        "division corrects only for the pattern's own catalogued "
        "fraction, not for those two."),
    ("density_pahn_region", "DENSITY_PAHN_CATALOGUED"): ("objects per square degree",
        "For each region (rows, REGION), one value per its own MERGED "
        "bin (MERGED_LOG10_LIM_EDGES): the MEASURED sky density of the "
        "PAHN pattern -- a catalogued source with a measured 5.8 and 8 "
        "micron flux and no measured 3.6 or 4.5 micron flux -- among "
        "this region's own admitted pixels whose own 8 micron "
        "50%-completeness limit falls in this merged bin: the total "
        "pattern count there divided by the total covered area there, "
        "with no correction for any source's own chance of being "
        "catalogued. A merged bin still at exactly zero is filled by "
        "log-linear interpolation if interior, held as measured at "
        "either end (DEFECT 2's own residual case, same rule as "
        "DENSITY_PAHN). An upper bound: the pattern count also contains "
        "any true protostar too faint at 3.6 and 4.5 to detect, and "
        "artefacts."),
    ("density_pahn_region", "N_PER_BIN"): ("sources",
        "For each region, one value per its own MERGED bin "
        "(MERGED_LOG10_LIM_EDGES): the total RAW pattern count "
        "DENSITY_PAHN_CATALOGUED in that merged bin rests on, before any "
        "log-linear interpolation (so a 0 here with a nonzero DENSITY_"
        "PAHN_CATALOGUED means that merged bin's own density was "
        "interpolated, DEFECT 2's own residual case). Every merged bin "
        "holds at least PAHN_MIN_PER_BIN (10) unless the region's own "
        "whole reached span holds fewer pattern sources than that in "
        "total. An upper bound: the pattern count also contains any "
        "true protostar too faint at 3.6 and 4.5 to detect, and "
        "artefacts."),
    ("density_pahn_region", "POISSON_LO"): ("objects per square degree",
        "The lower edge of the standard 68% (Garwood) Poisson confidence "
        "interval on DENSITY_PAHN_CATALOGUED, from N_PER_BIN's own "
        "merged-bin raw count, on the same density scale (computed fresh "
        "on the merged count, never averaged from the raw bins' own "
        "intervals)."),
    ("density_pahn_region", "POISSON_HI"): ("objects per square degree",
        "The upper edge of the standard 68% (Garwood) Poisson confidence "
        "interval on DENSITY_PAHN_CATALOGUED, from N_PER_BIN's own "
        "merged-bin raw count, on the same density scale (computed fresh "
        "on the merged count, never averaged from the raw bins' own "
        "intervals)."),
    ("density_pahn_region", "AREA_DEG2_PER_BIN"): ("square degrees",
        "For each region, one value per its own MERGED bin "
        "(MERGED_LOG10_LIM_EDGES): the total covered area (the any-IRAC "
        "union coverage fraction times the nside-512 pixel area, summed "
        "over every admitted pixel in that merged bin) both DENSITY_PAHN "
        "and DENSITY_PAHN_CATALOGUED divide their own numerator by."),
    ("density_pahn_region", "PATTERN_COUNT_REGION"): ("sources",
        "For each region, the total pattern count over every admitted "
        "pixel, before binning: N_PER_BIN's own row sum. An upper "
        "bound: the pattern count also contains any true protostar too "
        "faint at 3.6 and 4.5 to detect, and artefacts."),

    # population.pahn -- weight_pahn_source: one row per PAHN pattern
    # source survey-wide, its own probability of the catalogued event
    # (5.8 and 8 micron both detected) and the inverse weight DENSITY_PAHN
    # sums into its bins (owner's 2026-10-22 ruling, FINDINGS.md S07),
    # for pattern_f45_1kpc's own downstream consumer (the PAHN brightness
    # marginal) to weight by the identical quantity.
    ("weight_pahn_source", "NAME"): ("name",
        "The catalogue NAME of this PAHN pattern source (a catalogued "
        "source with a measured 5.8 and 8 micron flux and no measured "
        "3.6 or 4.5 micron flux)."),
    ("weight_pahn_source", "REGION"): ("name",
        "The region this pattern source belongs to."),
    ("weight_pahn_source", "LOG10_F_I4_MJY"): ("log10 mJy",
        "This pattern source's own measured 8 micron flux, as observed "
        "(not placed at any reference distance) -- the SAME flux "
        "pattern_f45_1kpc places at the register's 1 kpc convention."),
    ("weight_pahn_source", "P_CAT"): ("probability",
        "This pattern source's own probability of the catalogued event "
        "-- its own 5.8 and 8 micron flux both detected (its 3.6 and "
        "4.5 micron flux and both 2MASS and 24 micron flux are zero by "
        "the pattern's own definition, so they do not enter this "
        "probability): the survey's detection-probability mixture "
        "(independent, the product of each band's own completeness; "
        "comonotone, the smaller of the two; the measured LAMBDA share "
        "between them) at this source's own pixel and its own measured "
        "flux, floored at 1 minus the measurement's own completeness "
        "cap (about 1/200) so WEIGHT below is never a division by "
        "zero. This is selection into the pattern itself, not a general "
        "completeness for a star-less PAH nebular source: the pattern's "
        "own contamination (a true protostar too faint at 3.6 and 4.5 "
        "to detect; artefacts) and incompleteness (true PAHN outside "
        "the pattern) are a separate, undivided-out, two-directional "
        "upper-bound disclosure, carried by DENSITY_PAHN/DENSITY_PAHN_"
        "CATALOGUED alike."),
    ("weight_pahn_source", "WEIGHT"): ("sources per pattern source",
        "1 / P_CAT: this pattern source's own share of DENSITY_PAHN's "
        "numerator in whichever bin its own pixel's 8 micron 50%-"
        "completeness limit falls in (DENSITY_PAHN sums this over a "
        "bin's own pattern sources and divides by the bin's own total "
        "covered area) -- always >= 1, never larger than about 200."),

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
        "package's column_adopted_source A_COL_K, re-derived here), "
        "converted onto the scale of the Juvela & Montillaud (2016) "
        "near-infrared star-color extinction map by this region's one "
        "straight-line calibration (CAL_OFFSET_K plus CAL_SLOPE times "
        "the dust column), fitted the way Lombardi, Bouy, Alves and Lada "
        "(2014) calibrated a dust-emission map against the same kind of "
        "star-color map."),
    ("extinction_adopted_source", "A_COL_SIG_K"): ("mag A_K",
        "The adopted dust column's own 1-sigma uncertainty, scaled by "
        "this region's CAL_SLOPE."),
    ("extinction_adopted_source", "A_COL_PROVENANCE"): ("code",
        "Which arm the dust column A_COL_K is converted from, before "
        "this region's straight-line calibration: 0 Herschel, 1 Planck."),
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
    ("extinction_adopted_source", "CAL_OFFSET_K"): ("mag A_K",
        "One number for the whole region: the additive offset of the "
        "straight line that converts the dust-emission extinction column "
        "onto the scale of the Juvela & Montillaud (2016) near-infrared "
        "star-color extinction map, A_K(star-color) = CAL_OFFSET_K plus "
        "CAL_SLOPE times A_K(dust emission). Fitted the way Lombardi, "
        "Bouy, Alves and Lada (2014) calibrated a dust-emission map "
        "against a star-color map, over the region's sightlines where "
        "both maps read below 1 magnitude of K-band extinction, then "
        "applied to every source and sightline of the region."),
    ("extinction_adopted_source", "CAL_SLOPE"): ("dimensionless",
        "One number for the whole region: the multiplicative slope of "
        "the straight line that converts the dust-emission extinction "
        "column onto the scale of the Juvela & Montillaud (2016) "
        "near-infrared star-color extinction map, A_K(star-color) = "
        "CAL_OFFSET_K plus CAL_SLOPE times A_K(dust emission). Fitted "
        "the way Lombardi, Bouy, Alves and Lada (2014) calibrated a "
        "dust-emission map against a star-color map, over the region's "
        "sightlines where both maps read below 1 magnitude of K-band "
        "extinction, then applied to every source and sightline of the "
        "region."),

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
        "column (column_adopted_sightline A_K), converted onto the "
        "near-infrared star-color extinction scale by that pixel's own "
        "region's straight-line calibration (extinction_adopted_source's "
        "CAL_OFFSET_K and CAL_SLOPE)."),
    ("extinction_adopted_sightline", "SIGMA_A_K"): ("mag A_K",
        "The adopted sightline column's own 1-sigma uncertainty, scaled "
        "by that pixel's own region's CAL_SLOPE."),
    ("extinction_adopted_sightline", "PROVENANCE"): ("code",
        "Which arm the dust column came from before the region's "
        "straight-line calibration, carried unchanged from the adopted "
        "sightline column: 0 Herschel, 1 Planck."),

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

    # sky.derived.yso_census -- census_yso_census_survey: every published
    # excess-selected (and one protostar-only) young-star census the young-
    # star law is fit against (owner's 2026-10-20 ruling), one row per
    # catalogued object, replacing the retired yso_dunham2015_survey
    # (Dunham et al. 2015 is now one SOURCE among eight).
    ("census_yso_census_survey", "SOURCE"): ("census name",
        "Which published census this object comes from (e.g. 'Dunham2015', "
        "'Megeath2012', 'Strafella2010') -- see the module docstring of "
        "sesnaimpute.sky.derived.yso_census for the paper behind each name."),
    ("census_yso_census_survey", "CLOUD"): ("cloud name",
        "The star-forming cloud this object belongs to, in its own "
        "census's own naming (several census names share one CLOUD across "
        "several SOURCE values is possible, e.g. none here do; one SOURCE "
        "can carry several CLOUD values, e.g. Dunham2015's eighteen "
        "clouds or Megeath2012's Orion A / Orion B)."),
    ("census_yso_census_survey", "RA_DEG"): ("deg",
        "Right ascension, equinox J2000."),
    ("census_yso_census_survey", "DEC_DEG"): ("deg",
        "Declination, equinox J2000."),
    ("census_yso_census_survey", "DIST_PC"): ("pc",
        "The adopted distance to this object's own cloud: the census "
        "paper's own per-cloud distance where it gives one (Dunham2015 "
        "only), else this project's own adopted distance for the same "
        "cloud (sesnaimpute.regions.REGIONS_BY_NAME)."),
    ("census_yso_census_survey", "ALPHA"): ("dimensionless",
        "This object's infrared spectral index (d log(lambda F_lambda) / "
        "d log(lambda)), the census paper's own measurement where it "
        "gives one as a number; NaN where the census classifies the "
        "object by a discrete class label only, with no numeric slope."),
    ("census_yso_census_survey", "F45_MJY"): ("mJy",
        "This object's 4.5 micron flux density: the census paper's own "
        "measured flux where it gives one directly, else converted from "
        "its own 4.5 micron magnitude at the IRAC channel-2 zero point "
        "(179.7 Jy, Reach et al. 2005); NaN where the census gives "
        "neither. Not extinction-corrected except for Dunham2015, whose "
        "own Table 4 already is."),
    ("census_yso_census_survey", "LOG10_F45_REF"): ("log10 mJy at 1 kpc",
        "log10 of F45_MJY scaled to a common reference distance of 1 kpc "
        "by (DIST_PC / 1000)^2: 10**x is a flux in mJy a source at this "
        "object's own brightness would show at 1 kpc, the same reference-"
        "brightness scale and distance the YSO templates used elsewhere "
        "in this package are tabulated at. NaN wherever F45_MJY or "
        "DIST_PC is NaN."),
    ("census_yso_census_survey", "SELECTION"): ("selection kind",
        "'excess' for an excess-selected young star (the law's own kind: "
        "Class 0, I, flat, II, transition disk or weak excess), "
        "'protostar' for a protostar-only census (Kryukova2014/Cygnus X): "
        "the law unit decides how a protostar-only census enters a "
        "region's fit."),
    ("census_yso_census_survey", "SURVEY_FLOOR_A_K"): ("mag A_K",
        "This object's own census's coverage/completeness column floor: "
        "Dunham2015's c2d (0.22 mag) or Gould Belt (0.33 mag) rule; NaN "
        "for every other census here, which states no comparable "
        "extinction/coverage completeness criterion -- population.yso_law's "
        "fit footprint for such a census is the coverage polygon alone, "
        "with no column-floor restriction."),

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

    # sky.derived.profile -- profile_adopted_sightline (C26, owner ruling
    # 2026-10-21, briefs/round4_2026-10-20/C26_profile_adopted.md): the ONE
    # canonical cumulative K-band extinction curve per admitted sightline,
    # on the 2MASS stellar scale, every consumer now reads. Edenhofer et al.
    # 2023's own shape inside their 3-D map's edge, rescaled to this
    # sightline's own adopted extinction column; beyond the map's edge,
    # Combined19's (Bovy et al. 2016, mwdust) own measured INCREMENT --
    # Marshall et al. 2006 (2MASS red giants) where its own survey box
    # covers the sightline, Bayestar19 (Green et al. 2019) elsewhere it
    # reaches, Drimmel et al. 2003 only where neither exists. Replaces the
    # retired extinction_marshall_sightline product and the per-read
    # residual/tail-shape/scale algebra it fed (C25 and earlier): a reader
    # rescales the in-cube part by its own column over this file's
    # A_EDGE_K and adds the far part unscaled (sky.derived.profile.
    # _RegionProfile.a_of_d). C26b (owner ruling 2026-10-21b): the far
    # curve is the MEAN over a sightline's 16 nside-1024 child-pixel
    # centres, never a point lookup at the sightline's own centre (an
    # exact HEALPix hierarchy vertex, a genuine implementation-dependent
    # degeneracy C26's own identity 1 found); the in-cube amplitude is
    # partitioned against the far map's own measurement of the far part,
    # never capped at 1 (the retired module's down-branch is gone:
    # Edenhofer's own column always stands at least).
    ("profile_adopted_sightline", "HPX_PIX_256"): ("nested HEALPix pixel, nside 256",
        "This region's own admitted nside-256 pixel numbers (nested "
        "ordering). Row i here is row i of every other per-sightline "
        "dataset in this file. Each one is a WHOLE pixel, not a point: "
        "several datasets below (DIST_PC's far part, A_CUM_K's far "
        "part, FAR_MAP, FAR_RELIABLE_PC) are the MEAN or majority over "
        "the pixel's own 16 nside-1024 sub-pixels, not a single lookup "
        "at the pixel's own centre."),
    ("profile_adopted_sightline", "GAL_L_DEG"): ("deg",
        "Galactic longitude of the matching pixel's center in HPX_PIX_256."),
    ("profile_adopted_sightline", "GAL_B_DEG"): ("deg",
        "Galactic latitude of the matching pixel's center in HPX_PIX_256."),
    ("profile_adopted_sightline", "DIST_PC"): ("pc",
        "The distance grid along the line of sight, starting at 0 pc, "
        "that A_CUM_K is tabulated on for every sightline in this file: "
        "the Edenhofer 3-D map's own in-cube points out to its edge "
        "(D_EDGE_PC), then a log-spaced grid out to Combined19's own "
        "~10,000 pc reach."),
    ("profile_adopted_sightline", "A_CUM_K"): ("mag A_K",
        "For the matching pixel in HPX_PIX_256 and each distance in "
        "DIST_PC, this sightline's own cumulative K-band extinction from "
        "the observer out to that distance. Inside the Edenhofer map's "
        "own edge (distance <= D_EDGE_PC): the map's own measured shape, "
        "scaled so its value AT the edge equals A_EDGE_K -- A_EDGE_K is "
        "never below the map's own raw (unscaled) column there, only at "
        "or above it (see A_EDGE_K's own reading for the rule that sets "
        "it). Beyond the edge: A_EDGE_K plus Combined19's own measured "
        "INCREMENT past the edge along this same sightline's 16 child "
        "pixels' own mean curve -- never Combined19's own absolute "
        "value, and never capped at any column: a reader's own total "
        "column, divided into this number, can exceed 1 out here (the "
        "far reddening really is more than a near pixel's own cloud "
        "column implies)."),
    ("profile_adopted_sightline", "A_EDGE_K"): ("mag A_K",
        "For the matching pixel in HPX_PIX_256, A_CUM_K's own value at "
        "D_EDGE_PC: the LARGER of (a) the Edenhofer map's own raw "
        "(unscaled) column at its edge and (b) this sightline's own "
        "adopted extinction column (the catalogue's own extinction_"
        "adopted_sightline product) minus the far map's own measured "
        "reach beyond the edge (IN_CUBE_EXCESS_K's own construction) -- "
        "the owner's precedence rule (the emission column over the "
        "cube's own integral), with the far map's own measurement "
        "deciding how much of any excess the cube must carry versus the "
        "far field already accounts for. A reader rescales the in-cube "
        "part of A_CUM_K by its own total column divided by this "
        "number, capped at 1 (so u <= 1 inside the cube for that "
        "reader's own column)."),
    ("profile_adopted_sightline", "D_EDGE_PC"): ("pc",
        "The Edenhofer 3-D map's own edge distance, ONE number shared by "
        "every sightline of every region (the map's own fixed radial "
        "grid): the distance A_CUM_K's in-cube, Edenhofer-measured part "
        "ends and Combined19's own far increment begins."),
    ("profile_adopted_sightline", "IN_CUBE_EXCESS_K"): ("mag A_K",
        "For the matching pixel in HPX_PIX_256, A_EDGE_K minus the "
        "Edenhofer map's own raw (unscaled) column at the cube's edge: "
        "how much of this sightline's own adopted extinction column was "
        "placed INSIDE the cube because the far map's own measurement "
        "beyond the edge did not account for it all (owner ruling "
        "2026-10-21b) -- zero where the far map alone already covers "
        "the sightline's own excess (typically the Galactic plane, "
        "where Marshall or Bayestar measures the far column directly), "
        "positive at high Galactic latitude where the far map goes "
        "flat well short of the sightline's own total column."),
    ("profile_adopted_sightline", "FAR_MAP"): ("code",
        "For the matching pixel in HPX_PIX_256, which of Combined19's "
        "own three constituent surveys supplies the far column beyond "
        "D_EDGE_PC along this sightline: the code the MAJORITY of its "
        "16 nside-1024 child pixels share (see FAR_MAP_MIXED where they "
        "do not all agree), by Combined19's own published build rule "
        "(computed from each child's own sky position and the "
        "Bayestar19 survey's own footprint, never guessed): 1 Marshall "
        "et al. 2006 (2MASS red giants, within 100 degrees of the "
        "Galactic centre in longitude and 10 degrees of the plane in "
        "latitude), 2 Bayestar19 (Green et al. 2019, wherever it has a "
        "measured pixel outside that box), 3 Drimmel et al. 2003 (an "
        "all-sky smooth dust-disc model, wherever neither of the above "
        "reaches)."),
    ("profile_adopted_sightline", "FAR_MAP_MIXED"): ("boolean",
        "For the matching pixel in HPX_PIX_256, True where this "
        "sightline's own 16 nside-1024 child pixels do not all share "
        "the same FAR_MAP code -- the sightline straddles a boundary of "
        "Combined19's own build rule (almost always Marshall's own box "
        "edge). FAR_MAP itself is still the majority code among the 16; "
        "this flag says that majority was not unanimous."),
    ("profile_adopted_sightline", "FAR_RELIABLE_PC"): ("pc",
        "For the matching pixel in HPX_PIX_256, the MEDIAN, over "
        "whichever of its 16 child pixels FAR_MAP's own majority code "
        "classifies Bayestar, of Bayestar19's own per-pixel "
        "reliable-distance limit (their own quoted maximum distance "
        "modulus with a converged fit, converted to a distance); NaN "
        "where FAR_MAP's majority is not Bayestar, or where none of the "
        "Bayestar-classified children carries such a limit in their "
        "published file."),

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

    # catalog.depths -- depths_sesna_region: the fitted 2MASS (J, H, Ks)
    # 50%-completeness detection curve per region. The five Spitzer bands'
    # own fit retired (COMPLETENESS_2026-10-20.md): catalog.completeness's
    # measured per-pixel maps replace it, so DELTA_DEX, SIGMA_DELTA_DEX,
    # W_MAG, FIT_RESIDUAL and SUBSTITUTED (all Spitzer-only) are gone, and
    # WIDTH_DEX/SIGMA_WIDTH_DEX carry the three 2MASS bands only.
    ("depths_sesna_region", "REGION"): ("region name",
        "The name of the region this row describes, in the fixed region "
        "order every region-axis product in this package shares."),
    ("depths_sesna_region", "F50_2MASS_MJY"): ("mJy",
        "For each of the three 2MASS bands (J, H, Ks, this dataset's own "
        "column order), this region's fitted 50%-completeness flux."),
    ("depths_sesna_region", "FIT_RESIDUAL_2MASS"): ("dimensionless",
        "For each of the three 2MASS bands (J, H, Ks, this dataset's own "
        "column order), the fitted detection curve's relative L1 distance "
        "from this region's own observed magnitude histogram, measured "
        "only past one magnitude brighter than the fitted 50% point."),
    ("depths_sesna_region", "WIDTH_DEX"): ("dex",
        "For each of the three 2MASS bands (J, H, Ks, this dataset's own "
        "column order), this region's effective detection roll-off width "
        "(the directly fitted magnitude-space width times 0.4): the "
        "quantity catalog.completeness's 2MASS CURVE_J/H/Ks tabulation "
        "reads, beside F50_2MASS_MJY."),

    # catalog.depths -- depths_sesna_hpx128: beside the region-wide fit
    # above, this region's own per-nside-128-pixel 2MASS fit (briefs/
    # round4_2026-10-20/C6_twomass_pixel_depths.md): one row per nside-128
    # NESTED pixel (catalog.depth_grid.admitted_pixels_512's own (lon,
    # lat) pair, GAL_L_DEG/GAL_B_DEG) that holds at least one catalogue
    # source, any band. F50_2MASS_MJY/WIDTH_DEX/FIT_RESIDUAL_2MASS always
    # carry a usable value -- this pixel's own fit where it clears every
    # gate in catalog/depths.py's own module docstring, the region's own
    # depths_sesna_region row copied in verbatim otherwise (FROM_REGIONAL
    # true there). catalog.completeness folds this product's rows
    # straight into completeness_sesna_source's own CURVE_J/H/Ks tables.
    ("depths_sesna_hpx128", "HPX_PIX_128"): ("nested HEALPix pixel, nside 128",
        "This region's own nside-128 pixel numbers (nested ordering), "
        "ascending: every pixel holding at least one catalogue source "
        "(any band). Row i here is row i of every other dataset in this "
        "file, and the row catalog.completeness's own CURVE_J/H/Ks tables "
        "copy for this pixel's children."),
    ("depths_sesna_hpx128", "F50_2MASS_MJY"): ("mJy",
        "For each of the three 2MASS bands (J, H, Ks, this dataset's own "
        "column order), this pixel's own fitted 50%-completeness flux "
        "where FROM_REGIONAL is false, this region's depths_sesna_region "
        "value copied in verbatim where it is true."),
    ("depths_sesna_hpx128", "WIDTH_DEX"): ("dex",
        "Same meaning as depths_sesna_region's own WIDTH_DEX, at this "
        "pixel's own fit (or the region's, where FROM_REGIONAL is true)."),
    ("depths_sesna_hpx128", "FIT_RESIDUAL_2MASS"): ("dimensionless",
        "Same meaning as depths_sesna_region's own FIT_RESIDUAL_2MASS, "
        "at this pixel's own fit (or the region's, where FROM_REGIONAL is "
        "true)."),
    ("depths_sesna_hpx128", "N_DETECTED"): ("count",
        "For each of the three 2MASS bands, how many ORIGIN_FNU == 1 "
        "(detected) catalogue sources fall in this pixel -- the number "
        "the >= 300 gate is checked against, recorded whether or not "
        "this pixel's own fit was used."),
    ("depths_sesna_hpx128", "FROM_REGIONAL"): ("boolean",
        "For each of the three 2MASS bands, true where this pixel's own "
        "fit failed at least one of the four gates in catalog/depths.py's "
        "own module docstring (fewer than 300 detections in that band; "
        "the fit did not converge; its own 50% point more than 1.0 mag "
        "from the region's; its own fitted width not strictly inside "
        "_W_BOUNDS) and F50_2MASS_MJY/WIDTH_DEX/FIT_RESIDUAL_2MASS carry "
        "the region's own depths_sesna_region row instead of this "
        "pixel's own fit."),


    # catalog.completeness -- completeness_sesna_source: Rob Gutermuth's
    # own measured completeness curves, per region (COMPLETENESS_2026-10-
    # 20.md "The reader: layout and kernel", amended by that file's own
    # "Corrections 2026-10-20"; replaces catalog.depth_grid's limit/width
    # role and limits_sesna_source/depth-grid_sesna_hpx512). CURVE_<band>,
    # SOURCE_ROW and MAG_GRID_<band> share one convention per band -- see
    # catalog/completeness.py's own module docstring for the exact
    # per-band canonical grid (Corrections item 1) and the completion rule
    # (item 2).
    ("completeness_sesna_source", "CURVE_I1"): ("completeness fraction x600, uint16",
        "One row per on-map 30-arcsec pixel of this region's own I1 (3.6 "
        "micron) fields, field-major order: the differential completeness "
        "fraction (0..1) at each of MAG_GRID_I1's own magnitudes, stored as "
        "the fraction times 600 (lossless: every MEASURED value is an exact "
        "multiple of 1/600; a COMPLETED plane, where COMPLETED_FROM_I1 is "
        "not NaN, is the band's own median roll-off template and is not "
        "itself guaranteed an exact multiple, but is stored the same way). "
        "catalog.limits.completeness's kernel divides by 600 once per "
        "evaluation; no other reader divides directly."),
    ("completeness_sesna_source", "CURVE_I2"): ("completeness fraction x600, uint16",
        "Same meaning as CURVE_I1, for I2 (4.5 micron) (MAG_GRID_I2, "
        "COMPLETED_FROM_I2)."),
    ("completeness_sesna_source", "CURVE_I3"): ("completeness fraction x600, uint16",
        "Same meaning as CURVE_I1, for I3 (5.8 micron) (MAG_GRID_I3, "
        "COMPLETED_FROM_I3)."),
    ("completeness_sesna_source", "CURVE_I4"): ("completeness fraction x600, uint16",
        "Same meaning as CURVE_I1, for I4 (8.0 micron) (MAG_GRID_I4, "
        "COMPLETED_FROM_I4)."),
    ("completeness_sesna_source", "CURVE_M1"): ("completeness fraction x600, uint16",
        "Same meaning as CURVE_I1, for M1 (24 micron) (MAG_GRID_M1, "
        "COMPLETED_FROM_M1)."),
    ("completeness_sesna_source", "COMPLETED_FROM_I1"): ("mag, Vega",
        "For each row of CURVE_I1, the magnitude of the FIRST plane that is "
        "not this pixel's own measurement but the band's own median "
        "roll-off template continued from its last measured value "
        "(Corrections 2026-10-20.md item 2: Rob's own cube for this pixel's "
        "field was cut off before reaching 0); NaN where the measured curve "
        "reached 0 on its own, nothing here completed."),
    ("completeness_sesna_source", "COMPLETED_FROM_I2"): ("mag, Vega",
        "Same meaning as COMPLETED_FROM_I1, for I2 (4.5 micron)."),
    ("completeness_sesna_source", "COMPLETED_FROM_I3"): ("mag, Vega",
        "Same meaning as COMPLETED_FROM_I1, for I3 (5.8 micron)."),
    ("completeness_sesna_source", "COMPLETED_FROM_I4"): ("mag, Vega",
        "Same meaning as COMPLETED_FROM_I1, for I4 (8.0 micron)."),
    ("completeness_sesna_source", "COMPLETED_FROM_M1"): ("mag, Vega",
        "Same meaning as COMPLETED_FROM_I1, for M1 (24 micron)."),
    ("completeness_sesna_source", "SOURCE_ROW"): ("row index into CURVE_<band>",
        "For every curated source and each of the eight bands, definitions."
        "BANDS' own order (J, H, Ks, I1, I2, I3, I4, M1 -- the same order "
        "fittp.sweep's BAND_KEYS uses, so a band's own position here IS "
        "its column there, no translation): the row of that band's own "
        "CURVE_<band> holding this source's own completeness curve. For "
        "I1/I2/I3/I4/M1, its nearest on-map 30-arcsec pixel, -1 where the "
        "source's own position is off every field's map in that band (no "
        "curve, no non-detection term). For J/H/Ks, the row of catalog."
        "depths' own nside-128 pixel this source falls in -- never -1 "
        "(2MASS carries no map footprint to fall off, COMPLETENESS_2026-"
        "10-20.md: 2MASS is all-sky)."),
    ("completeness_sesna_source", "PIXEL_CURVE_I1"): ("dimensionless (completeness fraction, 0..1)",
        "For each of this region's admitted nside-512 pixels (HPX_PIX_512, "
        "nested ordering, this file's own admitted-pixel axis), the I1 (3.6 "
        "micron) completeness curve at MAG_GRID_I1's own magnitudes, the "
        "mean over the on-map 30-arcsec completeness pixels inside this "
        "nside-512 pixel (256 sample points per pixel, each at its nearest "
        "30-arcsec pixel's CURVE_I1 row, averaged over the points that fall "
        "on a map); all zero where PIXEL_COVERED_FRAC_I1 is 0. This is the "
        "IRAC-UNION average: a pixel only partly covered by I1 within the "
        "four bands' own union (PIXEL_COVERED_FRAC_I1 < PIXEL_UNION_FRAC) "
        "reads a curve diluted toward zero by the union's off-I1 points, so "
        "it never reaches completeness 0.5 even where I1's own on-map area "
        "is fully complete. The catalogued-probability readers (fraction_"
        "pixel, pixel_fraction_params) use this union average AS STORED, "
        "unchanged; the 50%-limit readers (catalog.limits._crossing_mag_"
        "pixel/fifty_percent_pixel/local_width_dex_pixel) instead divide it "
        "back by PIXEL_COVERED_FRAC_I1 / PIXEL_UNION_FRAC (this pixel's own "
        "I1 on-map share WITHIN the union) to recover the on-map curve "
        "before finding the crossing -- see PIXEL_UNION_FRAC."),
    ("completeness_sesna_source", "PIXEL_CURVE_I2"): ("dimensionless (completeness fraction, 0..1)",
        "Same meaning as PIXEL_CURVE_I1, for I2 (4.5 micron)."),
    ("completeness_sesna_source", "PIXEL_CURVE_I3"): ("dimensionless (completeness fraction, 0..1)",
        "Same meaning as PIXEL_CURVE_I1, for I3 (5.8 micron)."),
    ("completeness_sesna_source", "PIXEL_CURVE_I4"): ("dimensionless (completeness fraction, 0..1)",
        "Same meaning as PIXEL_CURVE_I1, for I4 (8.0 micron)."),
    ("completeness_sesna_source", "PIXEL_CURVE_M1"): ("dimensionless (completeness fraction, 0..1)",
        "Same meaning as PIXEL_CURVE_I1, for M1 (24 micron)."),
    ("completeness_sesna_source", "PIXEL_COVERED_FRAC_I1"): ("dimensionless (area fraction, 0..1)",
        "For each admitted nside-512 pixel in HPX_PIX_512, the fraction of "
        "its own area that falls on an I1 (3.6 micron) completeness field's "
        "on-map footprint (the share of 256 sample points per pixel whose "
        "nearest 30-arcsec pixel is on a map; the same sampling as the "
        "coverage product's FRAC); 0.0 where the pixel falls off every I1 "
        "field's map (PIXEL_CURVE_I1 is all zero there)."),
    ("completeness_sesna_source", "PIXEL_COVERED_FRAC_I2"): ("dimensionless (area fraction, 0..1)",
        "Same meaning as PIXEL_COVERED_FRAC_I1, for I2 (4.5 micron)."),
    ("completeness_sesna_source", "PIXEL_COVERED_FRAC_I3"): ("dimensionless (area fraction, 0..1)",
        "Same meaning as PIXEL_COVERED_FRAC_I1, for I3 (5.8 micron)."),
    ("completeness_sesna_source", "PIXEL_COVERED_FRAC_I4"): ("dimensionless (area fraction, 0..1)",
        "Same meaning as PIXEL_COVERED_FRAC_I1, for I4 (8.0 micron)."),
    ("completeness_sesna_source", "PIXEL_COVERED_FRAC_M1"): ("dimensionless (area fraction, 0..1)",
        "Same meaning as PIXEL_COVERED_FRAC_I1, for M1 (24 micron)."),
    ("completeness_sesna_source", "MAG_GRID_I1"): ("mag, Vega",
        "The magnitudes CURVE_I1 and PIXEL_CURVE_I1's own column order are "
        "measured at: I1's own survey-wide canonical grid, 4.75..19.75 mag "
        "in 0.5-mag steps (Corrections 2026-10-20.md item 1, measured over "
        "every one of the survey's own I1 cubes); a field's own native grid "
        "is placed here by EXACT plane alignment, never resampled -- "
        "brightward of a field's own first plane, that field's own first "
        "value; faintward of its own last measured plane, the band's own "
        "template continuation (COMPLETED_FROM_I1) or that field's own "
        "measured 0."),
    ("completeness_sesna_source", "MAG_GRID_I2"): ("mag, Vega",
        "Same meaning as MAG_GRID_I1, for I2: 3.75..18.75 mag."),
    ("completeness_sesna_source", "MAG_GRID_I3"): ("mag, Vega",
        "Same meaning as MAG_GRID_I1, for I3: 1.25..16.75 mag."),
    ("completeness_sesna_source", "MAG_GRID_I4"): ("mag, Vega",
        "Same meaning as MAG_GRID_I1, for I4: 1.75..16.25 mag."),
    ("completeness_sesna_source", "MAG_GRID_M1"): ("mag, Vega",
        "Same meaning as MAG_GRID_I1, for M1: 0.25..12.25 mag."),
    ("completeness_sesna_source", "ENVELOPE_I1"): ("completeness fraction x600, uint16",
        "This region's own I1 (3.6 micron) ENVELOPE curve: at each of "
        "MAG_GRID_I1's own magnitudes, the MAXIMUM of CURVE_I1 over every "
        "one of this region's own on-map 30-arcsec pixels -- the deepest "
        "completeness anywhere in the region's own footprint, position- "
        "independent. Read by population.field_stars's retention test, "
        "since a synthetic field star carries no sky position of its own "
        "(SPEC_PRIORS.md section 1.5): a star is retained if this curve is "
        "nonzero at its own magnitude in at least two of eight bands, never "
        "a single pointing's own curve. Stored as the fraction times 600, "
        "the same convention as CURVE_I1."),
    ("completeness_sesna_source", "ENVELOPE_I2"): ("completeness fraction x600, uint16",
        "Same meaning as ENVELOPE_I1, for I2 (4.5 micron) (MAG_GRID_I2)."),
    ("completeness_sesna_source", "ENVELOPE_I3"): ("completeness fraction x600, uint16",
        "Same meaning as ENVELOPE_I1, for I3 (5.8 micron) (MAG_GRID_I3)."),
    ("completeness_sesna_source", "ENVELOPE_I4"): ("completeness fraction x600, uint16",
        "Same meaning as ENVELOPE_I1, for I4 (8.0 micron) (MAG_GRID_I4)."),
    ("completeness_sesna_source", "ENVELOPE_M1"): ("completeness fraction x600, uint16",
        "Same meaning as ENVELOPE_I1, for M1 (24 micron) (MAG_GRID_M1)."),
    ("completeness_sesna_source", "PIXEL_UNION_FRAC"): ("dimensionless (area fraction, 0..1)",
        "For each admitted nside-512 pixel in HPX_PIX_512, the IRAC-UNION "
        "on-map share of its own 256 sample points: the fraction of those "
        "points that fall on ANY of the four IRAC bands' (I1-I4) own field "
        "maps (I1 or I2 or I3 or I4), the same union PIXEL_CURVE_<band>'s "
        "own mean is taken over for I1/I2/I3/I4/M1; 0.0 where no IRAC band "
        "covers the pixel at all. Always >= PIXEL_COVERED_FRAC_<band> for "
        "each of I1-I4 (that band's own on-map share is a subset of the "
        "union's), equal to it wherever the four footprints coincide. One "
        "number per pixel, not per band -- M1's own curve shares this same "
        "union denominator too (module docstring: 'M1 last: the build "
        "masks it to the IRAC union the four have established'). Read by "
        "catalog.limits's 50%-limit readers (_crossing_mag_pixel/fifty_"
        "percent_pixel/local_width_dex_pixel) to divide PIXEL_COVERED_FRAC_"
        "<band> / PIXEL_UNION_FRAC back out of PIXEL_CURVE_<band> and "
        "recover the band's own on-map curve for a pixel only partly "
        "covered within the union (DEFECT 1, bms_review/briefs/"
        "round6_2026-10-22/U13_limit_coverage.md); not used for 2MASS "
        "(J/H/Ks), which carries no union-average dilution of its own "
        "(PIXEL_COVERED_FRAC_J/H/Ks is always 1.0, an all-sky survey)."),
    ("completeness_sesna_source", "HPX_PIX_512"): ("nested HEALPix pixel, nside 512",
        "This region's own admitted nside-512 pixel numbers (nested "
        "ordering), ascending -- the same admission rule as "
        "catalog.depth_grid.admitted_pixels_512 (every nside-512 child of "
        "a source-bearing nside-256 pixel). Row i here is row i of every "
        "PIXEL_CURVE_<band>/PIXEL_COVERED_FRAC_<band> dataset in this file."),
    ("completeness_sesna_source", "CURVE_J"): ("completeness fraction x600, uint16",
        "The SAME layout as CURVE_I1 (briefs/round4_2026-10-20/C6_twomass_"
        "pixel_depths.md item 2), at nside-128 pixel resolution instead of "
        "30-arcsec: one row per nside-128 pixel of the region that holds "
        "at least one catalogue source, the error-function roll-off from "
        "catalog.depths' own per-pixel fit of F50_2MASS_MJY/WIDTH_DEX (or "
        "the region's, where that fit failed a gate -- FROM_REGIONAL_J "
        "true there), tabulated on MAG_GRID_J's own shared fine (0.05-mag) "
        "grid and stored as the fraction times 600. A row's own curve is "
        "clamped to exactly 1 brightward and 0 faintward of its OWN six "
        "roll-off widths, even past the shared grid's own wider span."),
    ("completeness_sesna_source", "CURVE_H"): ("completeness fraction x600, uint16",
        "Same meaning as CURVE_J, for H (MAG_GRID_H, FROM_REGIONAL_H)."),
    ("completeness_sesna_source", "CURVE_Ks"): ("completeness fraction x600, uint16",
        "Same meaning as CURVE_J, for Ks (MAG_GRID_Ks, FROM_REGIONAL_Ks)."),
    ("completeness_sesna_source", "COMPLETED_FROM_J"): ("mag, Vega",
        "Always NaN: 2MASS's own tabulated erf curve is never cut off "
        "the way a measured Spitzer field cube can be (COMPLETENESS_2026-"
        "10-20.md's own template-completion rule applies to I1-M1 only); "
        "kept so a reader of CURVE_J can use the SAME completed_from "
        "convention as the five Spitzer bands without a band-specific "
        "branch."),
    ("completeness_sesna_source", "COMPLETED_FROM_H"): ("mag, Vega",
        "Same meaning as COMPLETED_FROM_J, for H: always NaN."),
    ("completeness_sesna_source", "COMPLETED_FROM_Ks"): ("mag, Vega",
        "Same meaning as COMPLETED_FROM_J, for Ks: always NaN."),
    ("completeness_sesna_source", "FROM_REGIONAL_J"): ("boolean",
        "One entry per CURVE_J row (one per nside-128 pixel): true where "
        "that pixel's own J-band fit failed one of catalog.depths' own "
        "four gates and CURVE_J's row is the region-wide depths_sesna_"
        "region fit instead of that pixel's own (catalog.depths' own "
        "depths_sesna_hpx128 FROM_REGIONAL column, carried over verbatim)."),
    ("completeness_sesna_source", "FROM_REGIONAL_H"): ("boolean",
        "Same meaning as FROM_REGIONAL_J, for H."),
    ("completeness_sesna_source", "FROM_REGIONAL_Ks"): ("boolean",
        "Same meaning as FROM_REGIONAL_J, for Ks."),
    ("completeness_sesna_source", "PIXEL_CURVE_J"): ("dimensionless (completeness fraction, 0..1)",
        "The SAME meaning as PIXEL_CURVE_I1: for each of this region's "
        "admitted nside-512 pixels (HPX_PIX_512), the J-band completeness "
        "curve at MAG_GRID_J's own magnitudes of that pixel's own PARENT "
        "nside-128 pixel (nested, pix512 >> 4) -- CURVE_J's own row at "
        "that parent, read directly, cast to a float fraction. Never all "
        "zero (2MASS is all-sky, PIXEL_COVERED_FRAC_J is always 1)."),
    ("completeness_sesna_source", "PIXEL_CURVE_H"): ("dimensionless (completeness fraction, 0..1)",
        "Same meaning as PIXEL_CURVE_J, for H."),
    ("completeness_sesna_source", "PIXEL_CURVE_Ks"): ("dimensionless (completeness fraction, 0..1)",
        "Same meaning as PIXEL_CURVE_J, for Ks."),
    ("completeness_sesna_source", "PIXEL_COVERED_FRAC_J"): ("boolean (0.0 or 1.0)",
        "Always 1.0: every admitted nside-512 pixel's own parent nside-128 "
        "pixel carries a J-band curve (its own fit or the region's, never "
        "neither) since 2MASS is all-sky, unlike the Spitzer bands' own "
        "field footprints."),
    ("completeness_sesna_source", "PIXEL_COVERED_FRAC_H"): ("boolean (0.0 or 1.0)",
        "Same meaning as PIXEL_COVERED_FRAC_J, for H: always 1.0."),
    ("completeness_sesna_source", "PIXEL_COVERED_FRAC_Ks"): ("boolean (0.0 or 1.0)",
        "Same meaning as PIXEL_COVERED_FRAC_J, for Ks: always 1.0."),
    ("completeness_sesna_source", "MAG_GRID_J"): ("mag, Vega",
        "The magnitudes CURVE_J and PIXEL_CURVE_J's own column order are "
        "tabulated at: ONE shared fine (0.05-mag) grid spanning every "
        "nside-128 pixel's own six roll-off widths either side of its own "
        "50% point (the union: the grid's first plane is the brightest "
        "pixel's own m50 - span, its last the faintest pixel's own m50 + "
        "span); a pixel narrower than the union reads exactly 1 brightward "
        "and 0 faintward of its own span (CURVE_J's own clamp)."),
    ("completeness_sesna_source", "MAG_GRID_H"): ("mag, Vega",
        "Same meaning as MAG_GRID_J, for H."),
    ("completeness_sesna_source", "MAG_GRID_Ks"): ("mag, Vega",
        "Same meaning as MAG_GRID_J, for Ks."),
    ("completeness_sesna_source", "ENVELOPE_J"): ("completeness fraction x600, uint16",
        "This region's own J-band ENVELOPE curve: the SAME meaning as "
        "ENVELOPE_I1, at each of MAG_GRID_J's own magnitudes the MAXIMUM "
        "of CURVE_J over every one of this region's own nside-128 pixels "
        "-- the deepest completeness anywhere in the region, position-"
        "independent, read by population.field_stars's retention test "
        "the same way as the five Spitzer bands."),
    ("completeness_sesna_source", "ENVELOPE_H"): ("completeness fraction x600, uint16",
        "Same meaning as ENVELOPE_J, for H (MAG_GRID_H)."),
    ("completeness_sesna_source", "ENVELOPE_Ks"): ("completeness fraction x600, uint16",
        "Same meaning as ENVELOPE_J, for Ks (MAG_GRID_Ks)."),

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

    # sky.derived.planck_source_column: folded into sky.derived.column's
    # merge (its own per-source product's only reader) -- no product of
    # its own, no registry entries.

    # sky.derived.edenhofer_samples -- profile-sigma-samples_edenhofer_sightline:
    # the across-sample uncertainty on the Edenhofer et al. (2024) extinction
    # profile's depth mark, from its 12 released posterior samples.
    ("profile-sigma-samples_edenhofer_sightline", "HPX_PIX_256"): ("nested HEALPix pixel, nside 256",
        "This region's own nside-256 pixel numbers (nested ordering). Row i "
        "here is row i of SIGMA_RATIO_SAMPLES."),
    ("profile-sigma-samples_edenhofer_sightline", "DIST_PC"): ("pc",
        "The distance grid along the line of sight that SIGMA_RATIO_SAMPLES "
        "is tabulated on, shared by every pixel in this file."),
    ("profile-sigma-samples_edenhofer_sightline", "SIGMA_RATIO_SAMPLES"): ("dimensionless",
        "For the matching pixel in HPX_PIX_256 and each distance in "
        "DIST_PC, the standard deviation, across the map's 12 released "
        "posterior samples, of the ratio of the cumulative extinction at "
        "that distance to the sample's own total extinction along the "
        "whole sightline. A star's own position along the sightline is "
        "set by this same ratio, so this is the sample uncertainty on "
        "that positioning quantity: smaller near the far end, where the "
        "ratio is pinned near 1 for every sample."),

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

    # sky.derived.juvela_extinction -- extinction_juvela_source: the
    # Juvela & Montillaud (2016) NICEST star-color extinction map,
    # converted to A_K, at every cataloged source's own position.
    ("extinction_juvela_source", "A_K"): ("mag A_K",
        "For every cataloged source of this region, the NICEST star-color "
        "extinction map's value at the source's own sky position, converted "
        "from the map's native A_J to A_K with the adopted diffuse "
        "extinction law's own A_J/A_K ratio."),
    ("extinction_juvela_source", "HPX_PIX_1024"): ("nested HEALPix pixel, nside 1024",
        "For every cataloged source of this region, the nested nside-1024 "
        "pixel number of the source's own sky position: the map cell this "
        "file's A_K value is read at."),

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

    # sky.derived.coverage -- coverage_spitzer_hpx512 (nside-512): the
    # Spitzer mosaic coverage fraction per pixel and band, per region.
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
    ("counts_twomass_hpx512", "CELL_KS_EDGES"): ("mag 2MASS Ks",
        "The edges of the five half-magnitude Ks bins N_CELL/CELL_COMPLETE "
        "are tabulated on, 12.0 to 14.3 -- the STAR anchor's own 2MASS-only "
        "range (C30, briefs/round4_2026-10-20/C30_population_anchor.md): "
        "bin k runs from CELL_KS_EDGES[k] to CELL_KS_EDGES[k+1]."),
    ("counts_twomass_hpx512", "CELL_JK_EDGES"): ("mag J-Ks (2MASS)",
        "The three interior J-Ks colour edges N_CELL/CELL_COMPLETE's own "
        "last axis is tabulated on (C30): bin 0 is J-Ks < CELL_JK_EDGES[0] "
        "(0.6), bins 1 and 2 the two intervals between, and bin 3 (the "
        "reddest) is J-Ks > CELL_JK_EDGES[-1] (1.5) -- a 2MASS row with no "
        "usable J (ph_qual's J character 'U' or 'X', or no J measurement "
        "at all) is counted in this reddest bin, never assumed blue."),
    ("counts_twomass_hpx512", "N_CELL"): ("stars",
        "For the matching pixel in HPX_PIX_512, each Ks bin of "
        "CELL_KS_EDGES and each J-Ks colour bin of CELL_JK_EDGES, how many "
        "2MASS Point Source Catalog detections in Ks 12.0-14.3 fall in "
        "that pixel, Ks bin and colour bin (C30) -- the SAME rows N counts "
        "in its own four Ks bins over this range, split further by J-Ks; "
        "read by population.anchor_tiles/anchor_weights to fit a separate "
        "weight for the model's giant and dwarf populations, since "
        "TRILEGAL's own error in the Galactic plane is per population, "
        "not per Ks magnitude alone."),
    ("counts_twomass_hpx512", "CELL_COMPLETE"): ("boolean",
        "For the matching pixel in HPX_PIX_512, each Ks bin of "
        "CELL_KS_EDGES and each J-Ks colour bin of CELL_JK_EDGES, True "
        "where this pixel's own 2MASS J 50%-completeness magnitude "
        "(catalog.depths' per-pixel product) clears the cell's own "
        "faintest possible J -- the Ks bin's own faint edge plus the "
        "colour bin's own faint edge, the reddest bin standing in its own "
        "last finite edge, 1.5 -- by 0.3 mag (C24e's own margin, "
        "c18_diag/c24e_colour_cells.py) -- read by nothing downstream "
        "(every cell in the five C30 regions measured 2026-10-21 was "
        "already complete), kept because it is a measured fact about the "
        "pixel, not an assumption the fit silently makes."),

    ("counts_ukidss_hpx512", "HPX_PIX_512"): ("nested HEALPix pixel, nside 512",
        "The region's own occupied nside-512 pixel numbers (nested ordering), "
        "ascending. Row i here is row i of N."),
    ("counts_ukidss_hpx512", "MAG_EDGES"): ("mag UKIRT K",
        "The edges of the half-magnitude UKIDSS GPS K bins (UKIRT "
        "photometric system) this file counts stars in, 9.0 to 18.5 (C28, "
        "2026-10-21: raised from 17.0, a half-bin of headroom past the "
        "18.0 anchor ceiling): bin j runs from MAG_EDGES[j] to "
        "MAG_EDGES[j+1]."),
    ("counts_ukidss_hpx512", "N"): ("stars",
        "For the matching pixel in HPX_PIX_512 and each magnitude bin in "
        "MAG_EDGES, how many point-like (mergedClass -1 or -2) UKIDSS GPS "
        "detections fall in that pixel and bin, with no cut on the "
        "archive's own error-bits column (C19, 2026-10-21: the download's "
        "own former error-bit and narrow-class cut undercounted real stars "
        "by a measured 21-23% at Ks 12-14)."),
    ("counts_ukidss_hpx512", "CALIBRATION"): ("dimensionless",
        "Three numbers, this region's own overlap-window ratio and its "
        "combined-Poisson interval: [0] the ratio of 2MASS's own point "
        "count to this file's own N, summed over the Ks bins from "
        "OVERLAP_LOWER_MAG (13.0) to the 2MASS anchor's own faint edge "
        "(14.3) and over the pixels this file's N shows any UKIDSS "
        "coverage in at all; [1] and [2] that ratio's lower and upper "
        "bound under combined Poisson shot noise on both counts. All "
        "three NaN where this region's download carries no UKIDSS rows at "
        "all. population.anchor_tiles multiplies this file's own deep (Ks "
        "past 14.0) counts by CALIBRATION[0] before anything else reads "
        "them -- 2MASS is the reference survey, UKIDSS the extension, and "
        "no SESNA count enters this ratio on either side."),
    ("counts_ukidss_hpx512", "KS_COMPLETE_TO"): ("mag UKIRT K",
        "C28, 2026-10-21: this region's own MEASURED completeness turnover "
        "-- the faint edge of the half-magnitude bin whose own "
        "deep-covered-pixel differential count SLOPE first falls below "
        "the brighter bins' own fitted slope trend by more than one "
        "Poisson sigma (sky.derived.ukidss_counts.differential_turnover_"
        "mag), already rounded down to the half-magnitude grid. Where a "
        "LATER bin's own count exceeds the flagged bin's (the flag did "
        "not hold -- the brief's own ambiguous case), this is instead "
        "PUBLISHED_DEPTH_MAG (18.0, Lucas et al. 2008, MNRAS 391, 136), "
        "never a silently patched measurement. This file's own faint "
        "edge (18.5) if no bin's slope ever falls below the trend within "
        "the measured range, never beyond it. NaN where this region "
        "carries no UKIDSS coverage at all. population.anchor_tiles "
        "takes the region's own deep anchor edge as min(18.0, this "
        "number), never past the survey's own measured depth or its "
        "published ceiling."),

    # sky.derived.vhs_counts -- counts_vhs_hpx512: C19 Part B's VISTA
    # Hemisphere Survey sibling of counts_ukidss_hpx512, read where
    # UKIDSS does not cover a pixel.
    ("counts_vhs_hpx512", "HPX_PIX_512"): ("nested HEALPix pixel, nside 512",
        "The region's own occupied nside-512 pixel numbers (nested ordering), "
        "ascending. Row i here is row i of N."),
    ("counts_vhs_hpx512", "MAG_EDGES"): ("mag VISTA Ks",
        "The edges of the half-magnitude VHS Ks bins (VISTA photometric "
        "system) this file counts stars in, 9.0 to 18.5 (C28, 2026-10-21: "
        "raised from 17.0, a half-bin of headroom past the 18.0 anchor "
        "ceiling): bin j runs from MAG_EDGES[j] to MAG_EDGES[j+1]."),
    ("counts_vhs_hpx512", "N"): ("stars",
        "For the matching pixel in HPX_PIX_512 and each magnitude bin in "
        "MAG_EDGES, how many point-like (mergedClass -1 or -2) VHS "
        "detections fall in that pixel and bin, with no cut on the "
        "archive's own error-bits column."),
    ("counts_vhs_hpx512", "CALIBRATION"): ("dimensionless",
        "Three numbers, this region's own overlap-window ratio and its "
        "combined-Poisson interval: [0] the ratio of 2MASS's own point "
        "count to this file's own N, summed over the Ks bins from "
        "OVERLAP_LOWER_MAG (13.0) to the 2MASS anchor's own faint edge "
        "(14.3) and over the pixels this file's N shows any VHS coverage "
        "in at all; [1] and [2] that ratio's lower and upper bound under "
        "combined Poisson shot noise on both counts. All three NaN where "
        "this region's download carries no VHS rows at all. population."
        "anchor_tiles multiplies this file's own deep (Ks past 14.0) "
        "counts by CALIBRATION[0] before anything else reads them -- "
        "2MASS is the reference survey, VHS the extension, and no SESNA "
        "count enters this ratio on either side."),
    ("counts_vhs_hpx512", "KS_COMPLETE_TO"): ("mag VISTA Ks",
        "C28, 2026-10-21: this region's own MEASURED completeness turnover "
        "-- the faint edge of the half-magnitude bin whose own "
        "deep-covered-pixel differential count SLOPE first falls below "
        "the brighter bins' own fitted slope trend by more than one "
        "Poisson sigma (sky.derived.vhs_counts.differential_turnover_mag, "
        "the same construction sky.derived.ukidss_counts uses), already "
        "rounded down to the half-magnitude grid. Where a LATER bin's own "
        "count exceeds the flagged bin's (the flag did not hold -- the "
        "brief's own ambiguous case), this is instead PUBLISHED_DEPTH_MAG "
        "(18.0, McMahon et al. 2013, The Messenger 154, 35), never a "
        "silently patched measurement. This file's own faint edge (18.5) "
        "if no bin's slope ever falls below the trend within the "
        "measured range, never beyond it. NaN where this region carries "
        "no VHS coverage at all. population.anchor_tiles takes the "
        "region's own deep anchor edge as min(18.0, this number), never "
        "past the survey's own measured depth or its published ceiling."),

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
        "nebula-contaminated aperture, PAH nebular source (no star in the "
        "aperture), background galaxy, young stellar object, shocked gas knot. "
        "Column order is the CLASSES attribute. Sums to 1."),
    ("posterior_classification_source", "P_SUBCLASS"): ("probability",
        "Probability of each subclass in the SUBCLASSES attribute (every subclass "
        "in definitions.CLASSMAP), named class and subclass: field stars by "
        "spectral type O to T, evolved stars oxygen-rich or carbon-rich, galaxies "
        "active, star-forming, composite or passive, young stellar objects class "
        "0 to III or transition disk, knots J, C, steady C or C-J shocks. A "
        "class's columns sum to its P_CLASS; the whole row sums to 1."),
    ("posterior_classification_source", "LOG_ODDS"): ("nats",
        "How strongly the pipeline favours each class over all the others combined, "
        "as the natural log of the odds P/(1-P). One column per class, CLASSES "
        "order. 0 is even odds; +3 about 20 to 1 for; -3 about 20 to 1 against."),
    ("posterior_classification_source", "LOG_ODDS_PRIOR"): ("nats",
        "The part of LOG_ODDS that comes from the prior: from the prior atlas at "
        "this source's sky pixel (the pixel's mean column, not the source's own), "
        "the prior odds of this class against the other six, before this "
        "source's own measurements are used. LOG_ODDS_PRIOR + LOG_ODDS_DATA = "
        "LOG_ODDS. Not a number for a class whose models carry no flux in any band "
        "observed at this pixel (galaxies and shocked-gas knots outside the Spitzer "
        "IRAC coverage): the class cannot be catalogued there under the model as "
        "built, which is not a defect."),
    ("posterior_classification_source", "LOG_ODDS_DATA"): ("nats",
        "The part of LOG_ODDS that comes from this source itself: the photometry's "
        "Bayes factor against the prior atlas at this source's sky pixel, how well "
        "each class's population of model spectra, with the prior's spread in "
        "brightness and extinction, explains its measured fluxes, its "
        "non-detections and its Gaia measurement. Positive means the source's own "
        "data favour the class. Not a number where LOG_ODDS_PRIOR is (see there)."),
    ("posterior_classification_source", "LOG_ODDS_MEASURED"): ("nats",
        "The part of LOG_ODDS_DATA that comes from the bands the source was measured "
        "in: how much they move each class's log odds, measured against the prior "
        "atlas at this source's sky pixel. "
        "LOG_ODDS_MEASURED + LOG_ODDS_NONDET + LOG_ODDS_GAIA = LOG_ODDS_DATA. Not a "
        "number where the fit does not record it."),
    ("posterior_classification_source", "LOG_ODDS_NONDET"): ("nats",
        "The part of LOG_ODDS_DATA that comes from the bands the survey observed but "
        "did not detect the source in, after the measured bands: how much they "
        "move each class's log odds. A non-detection can only count against a class "
        "on its own; it raises a class's odds only by counting more heavily against "
        "the others. Not a number where the fit does not record it."),
    ("posterior_classification_source", "LOG_ODDS_GAIA"): ("nats",
        "The part of LOG_ODDS_DATA that comes from Gaia, after the measured and "
        "undetected bands: how much it moves each class's log odds. Gaia DR3's "
        "published completeness at the source's sky pixel; an unmatched source "
        "is scored as a non-detection, not as nothing. A matched source's chance "
        "of an unrelated Gaia star landing nearby by coincidence is scored "
        "against the region's simulated field-star population's own parallax "
        "distribution, standing in for the Gaia-detected chance-counterpart "
        "population itself, which the survey does not separately catalogue. Not "
        "a number where the fit does not record it."),
    ("posterior_classification_source", "P_YSO"): ("probability",
        "Probability that the source is a young stellar object: the young stellar "
        "object column of P_CLASS."),
    ("posterior_classification_source", "ENTROPY_CLASS"): ("fraction of maximum",
        "How spread the seven class probabilities are, divided by their maximum "
        "spread. 0 means one class takes all the probability, 1 means all seven are "
        "equally likely."),
    ("posterior_classification_source", "ENTROPY_SUBCLASS"): ("fraction of maximum",
        "How spread the subclass probabilities are (every subclass in "
        "definitions.CLASSMAP), divided by their maximum spread. 0 means one "
        "subclass takes all the probability, 1 means every subclass is equally "
        "likely."),
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
        "bands it is zero. This is the fit's own uncertainty: the prior enters only "
        "through the weight each model carries, never by moving a predicted flux, "
        "so it is not a credible interval over the true brightness. It carries no "
        "uncertainty between classes, since it is built from the one class the "
        "source was assigned to, and it is a local approximation at that class's "
        "own fixed weights, not an exact uncertainty under every reweighting."),
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
        "Four indicators, each casting one vote across the seven classes in CLASSES "
        "order: the class mix expected at the source's position, Gaia's detection "
        "and parallax, the Gutermuth color cuts on the measured fluxes, and those "
        "cuts on the filled-in spectrum. An indicator with nothing to say abstains, "
        "so a row sums to between 0 and 4. The PAH nebular source class has no star "
        "and so no rule in the Gutermuth color cuts (its own two bands, 3.6 and 4.5 "
        "micron, are exactly the two this class is defined by not having): the two "
        "color-cut indicators never vote for it, only the class-mix and Gaia "
        "indicators can. Reported only; the classification does not use it."),
    ("cascade_classification_source", "ENTROPY_PSI_VOTES"): ("fraction of maximum",
        "How spread those votes are, divided by their maximum spread. 0 means the "
        "indicators that voted agreed, 1 means they spread evenly over the seven "
        "classes. Not a number where none voted."),

    # fittp.library_resolution -- library-resolution_check_survey: whether
    # each class's library samples spectral shape finely enough, against
    # the library's own fixed sampling scale (twice the survey's
    # calibration precision, `constants.LIBRARY_SAMPLING_SIGMA_LOG_DEX`),
    # one survey-wide file (WP-PRIOR-8; moved from fittp/
    # library_resolution.py's own _READINGS, READINGS brief section 3).
    ("library-resolution_check_survey", "LIBRARY"): ("library name",
        "The model library this row describes, one per class."),
    ("library-resolution_check_survey", "BANDS"): ("band name",
        "The eight SESNA bands, in SIGMA_LIB_DEX's column order."),
    ("library-resolution_check_survey", "N_TEMPLATES"): ("models",
        "How many models that library holds."),
    ("library-resolution_check_survey", "D50"): ("sampling lengths",
        "Median distance from a model of that library to its nearest neighbor, in "
        "units of the library's own fixed sampling scale (twice the survey's "
        "calibration precision, per band, combined into one dimensionless shape "
        "space)."),
    ("library-resolution_check_survey", "D90"): ("sampling lengths",
        "The same distance at the ninetieth percentile: the spacing of that "
        "library's most isolated models. Reported alongside D50 but does not enter "
        "SIGMA_LIB_DEX."),
    ("library-resolution_check_survey", "SIGMA_LIB_DEX"): ("dex",
        "How finely that library samples spectral shape, per band, read straight off "
        "D50 (the formula alone, no fallback): the fit adds this in quadrature to "
        "that band's measurement error and its calibration systematic, so no model "
        "can beat a near-identical one by more than the sampling allows."),

}

# --- fittp.sweep -- <CLS>_fit_source (P7): one file per {region, class},
# the class evidence, imputed flux and top-K posterior record. The same
# dataset names and readings repeat in every one of the seven classes' own
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
        "How well this class explains the source, one value per subclass named "
        "in the SUBCLASSES attribute, as a natural logarithm. Each gathers every "
        "model of that subclass under both extinction laws, the measured and "
        "undetected bands and Gaia: Gaia DR3's published completeness at the "
        "source's sky pixel; an unmatched source is scored as a non-detection. "
        "It includes how wide a range of brightness and extinction the fit to "
        "the measured bands alone leaves plausible: a class whose measured-band "
        "errors are larger leaves a wider range and is marked down for it, "
        "separately from how well its single best-fitting value matches. "
        "Minus infinity means the subclass has no models, or the class cannot "
        "explain the source, or the source could not be fitted."),
    "LOG10_FLUX_MEAN": ("log10 of flux in mJy",
        "This class's estimate of the eight-band spectrum, band order J, H, Ks, "
        "3.6, 4.5, 5.8, 8.0 and 24 micron, averaging its models by their share of "
        "the support for the source. 10**x is a flux in mJy. The prior enters only "
        "through each model's share of the support, never by moving a model's own "
        "predicted flux, so this is the fit's own weighted prediction, not a "
        "posterior mean of the true brightness."),
    "LOG10_FLUX_COV": ("squared dex",
        "Covariance of those eight values, one dex being a factor of 10. It "
        "combines disagreement between this class's models with the precision of "
        "the fitted brightness and extinction. sqrt(diagonal) is the uncertainty in "
        "log flux. This is the fit's own uncertainty, with the prior entering only "
        "through the model weights, not a posterior credible interval; it is a "
        "local approximation at those weights' own fixed values, not an exact "
        "uncertainty under every reweighting."),
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
    "LN_EVIDENCE_MEASURED": ("nats",
        "This class's evidence from its prior and the measured bands only, as a "
        "natural logarithm, summed over every model in its library and over "
        "extinction; the undetected bands and Gaia are not yet included. It "
        "includes how wide a range of brightness and extinction the measured-"
        "band fit alone leaves plausible, carried through to LN_EVIDENCE_NONDET "
        "and LN_EVIDENCE as well. LN_EVIDENCE_MEASURED minus LN_PRIOR_MASS is "
        "how much the measured bands raise or lower the class."),
    "LN_EVIDENCE_NONDET": ("nats",
        "This class's evidence once the undetected bands are also included (Gaia "
        "not yet), as a natural logarithm summed over every model and over "
        "extinction. Each undetected but covered band's own chance of being "
        "missed reads SESNA's own measured completeness at the source's position: "
        "the fraction of injected sources of that brightness the survey recovered "
        "there, linear between the measurement's magnitude steps and continued "
        "along that band's typical falloff shape where the measurement itself "
        "stops, never letting a band be charged as more than 99.5 percent certain "
        "to have been missed. Bands observed at the same place share the same "
        "local background and confusion, so their own chances of being missed are "
        "not treated as fully separate: this blends treating them as separate "
        "with treating them as moving together (missed or not, as one), by one "
        "survey-wide share measured against real stars. Never above "
        "LN_EVIDENCE_MEASURED: a band the source was not detected in can only "
        "count against a class. The class's full evidence adds Gaia to this: "
        "Gaia DR3's published completeness at the source's sky pixel; an "
        "unmatched source is scored as a non-detection; a matched source's "
        "chance of an unrelated Gaia star landing nearby by coincidence is "
        "scored against the region's simulated field-star population's own "
        "parallax distribution, standing in for the Gaia-detected chance-"
        "counterpart population itself, which the survey does not separately "
        "catalogue."),
    "LN_PRIOR_MASS": ("natural log of objects per square degree",
        "The ln of the expected number per square degree of this class's objects "
        "the survey would catalogue at the source's sky pixel, from the prior "
        "atlas: the class's population at the pixel's mean column, each member "
        "weighted by its probability of clearing two of eight bands at the "
        "pixel's own completeness curves; the pixel's mean column, not the "
        "source's own. No photometry. Divided across the seven classes it gives "
        "each class's prior probability. Minus infinity for a class whose models "
        "carry no flux in any band observed at this pixel (galaxies and shocked-gas "
        "knots outside the Spitzer IRAC coverage): the class cannot be catalogued "
        "there under the model as built, which is not a defect."),
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
        "How strongly the prior and the measured bands together support each of "
        "those five models, as a natural logarithm, before the undetected bands "
        "and Gaia are considered. It is a single combined number, summed over the "
        "range of extinction the source allows; it cannot be split into a prior "
        "part and a fit part."),
    "TOPK_LN_UNSEEN": ("nats",
        "What the bands observed but not detected contributed, as a log factor "
        "given the prior and the measured fit, integrated over extinction; each "
        "such band's own contribution reads SESNA's own measured completeness at "
        "the source's position (the fraction of injected sources of that "
        "brightness the survey recovered there, linear between the measurement's "
        "magnitude steps and continued along that band's typical falloff shape "
        "where the measurement itself stops, never more than 99.5 percent certain "
        "a band was missed); reduces to the point value when the fit pins the "
        "extinction."),
    "TOPK_LN_GAIA": ("nats",
        "What Gaia contributed, as a log factor given the rest, integrated over "
        "extinction: Gaia DR3's published completeness at the source's sky "
        "pixel; an unmatched source is scored as a non-detection; a matched "
        "source's chance of an unrelated Gaia star landing nearby by "
        "coincidence is scored against the region's simulated field-star "
        "population's own parallax distribution, standing in for the "
        "Gaia-detected chance-counterpart population itself, which the survey "
        "does not separately catalogue."),
    "TOPK_LN_PRIOR_ML": ("nats",
        "The prior at the best-fit point; a guide to how TOPK_LN_PRIOR_FIT splits "
        "between the prior and the fit, not an addend."),
    "TOPK_LN_GAMMA_ML": ("nats",
        "The Gaia term at the best-fit point (Gaia DR3's published completeness "
        "at the source's sky pixel; an unmatched source is scored as a "
        "non-detection; a matched source's chance of an unrelated Gaia star "
        "landing nearby by coincidence is scored against the region's "
        "simulated field-star population's own parallax distribution, "
        "standing in for the Gaia-detected chance-counterpart population "
        "itself, which the survey does not separately catalogue); report-only; "
        "the vote reading uses it."),
    "TOPK_LAW": ("law position",
        "Which extinction law fits each of those five models better: 0 diffuse, 1 "
        "dense."),
}
for _cls in ("STAR", "AGB", "PAHC", "PAHN", "GAL", "YSO", "H2S"):
    _stem = "%s_fit_source" % _cls
    for _name, _rd in _SWEEP_FIT_READINGS.items():
        REGISTRY[(_stem, _name)] = _rd

# --- fittp.emission -- <lib>_emission_survey: the measured emission table
# E[k, b, v], one survey-wide file per library. The same dataset names and
# readings repeat in every one of the seven libraries' own files (moved from
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
for _lib in ("sps", "agb", "pahc", "galz", "yso", "h2shock", "pahn"):
    _stem = "%s_emission_survey" % _lib
    for _name, _rd in _EMISSION_READINGS.items():
        REGISTRY[(_stem, _name)] = _rd


# sky.derived.pacs_completeness -- completeness_pacs_source (2026-10-26 repair): the PACS 70 and 100
# micron completeness in the layout of completeness_sesna_source, plus per-cell columns.
_PACS_COMPLETENESS_CELLS_READINGS = {
    "CURVE_{b}": ("completeness fraction x600, uint16",
        "One row per covered 30-arcsec cell of every Herschel PACS {w} micron map, for all cells inside the "
        "region's source bounding box padded by about 7 arcmin (overlapping maps each keep their cells; "
        "SOURCE_ROW points at the deepest one): the probability that a point source of the flux in "
        "MAG_GRID_{b} placed anywhere in the cell is detected at 5 sigma or better, stored as the "
        "probability times 600. Measured by forced fits at {n} random positions per cell on the real map, "
        "so real background structure and neighbouring sources are included; sources fainter than the "
        "last plane are treated as undetected. Rows are ordered by map and then cell."),
    "MAG_GRID_{b}": ("mag relative to 1 Jy",
        "The magnitudes CURVE_{b} and PIXEL_CURVE_{b} are tabulated at: m = -2.5 log10(flux / 1 Jy), "
        "{mmin} to {mmax} in steps of {dm}, bright to faint (the same grid for every region). The "
        "zero point is 1 Jy, stored on CURVE_{b} as ZERO_POINT_JY."),
    "ENVELOPE_{b}": ("completeness fraction x600, uint16",
        "At each magnitude of MAG_GRID_{b}, the largest value of CURVE_{b} over all rows: the deepest "
        "{w} micron completeness anywhere in this region, stored as the probability times 600; all zero "
        "where the region has no {w} micron coverage."),
    "COMPLETED_FROM_{b}": ("mag",
        "All NaN: every {w} micron curve is measured over its whole range; nothing is extrapolated. Kept "
        "so the layout matches completeness_sesna_source."),
    "N_POS_{b}": ("positions",
        "For each row of CURVE_{b}, the number of random positions in the cell that lay on valid data "
        "and entered its curve and noise (at most {n})."),
    "N_SIGMA_POS_{b}": ("positions",
        "For each row of CURVE_{b}, how many random positions set its noise: the positions on valid data "
        "farther than one beam from every bright source, summed over the cells of the 3 x 3 block of "
        "cells around it whose noise entered the median. 0 means no cell of the block was clean and "
        "the map's median noise was used."),
    "CELL_RA_{b}": ("deg ICRS", "Right ascension of the centre of each cell of CURVE_{b}, in the same row order."),
    "CELL_DEC_{b}": ("deg ICRS", "Declination of the centre of each cell of CURVE_{b}, in the same row order."),
    "CELL_OBSID_{b}": ("observation identifier(s)",
        "The Herschel observation identifier(s), comma separated, of the map each cell of CURVE_{b} was "
        "measured on."),
    "CELL_MAP_{b}": ("map identifier",
        "The archive file name of the {w} micron map each cell of CURVE_{b} was measured on."),
    "CELL_SIGMA_{b}_MJY": ("mJy",
        "The one-sigma uncertainty of a {w} micron point-source flux measured in each cell of CURVE_{b}: "
        "the median, over the 3 x 3 block of cells around it, of 1.4826 times the median absolute deviation "
        "about CELL_OFFSET_{b}_MJY of the fluxes measured at each cell's random positions farther than one "
        "beam from every bright source (one cell alone holds only about ten independent beams), scaled by "
        "one factor per map so that negative excursions on empty sky are no more frequent than for Gaussian "
        "noise. A source is called detected at 5 times this value."),
    "CELL_OFFSET_{b}_MJY": ("mJy",
        "For each row of CURVE_{b}: the median flux the measurement returns at the cell's random positions farther "
        "than one beam from every bright source -- sky structure the annulus sky leaves at this scale. Detection of a "
        "source in the cell, and the completeness curve, count from this level; the reported flux (FNU in the "
        "photometry product, the total flux) does not subtract it, so the two differ by this value. A parallel-mode map that holds only one "
        "scan direction is measured with that direction's own point-spread template and aperture fraction (not at 100 micron)."),
    "CELL_TAIL_FACTOR_{b}": ("dimensionless",
        "For each row of CURVE_{b}: the factor by which the cell's noise was raised so that, among the cells of its "
        "map at a similar noise level, negative excursions on empty sky are no more frequent than for Gaussian noise, "
        "matched as far into the tail as those cells' sample allows (the map noise has heavier tails than a Gaussian, "
        "most in the noisiest cells). CELL_SIGMA_{b}_MJY already includes it."),
    "MAP_ID_{b}": ("map identifier",
        "One row per {w} micron map of the region: its archive file name. CELL_MAP_{b} names one of these for each cell."),
    "MAP_OBSID_{b}": ("observation identifier(s)",
        "For each row of MAP_ID_{b}: the Herschel observation identifier(s) the map combines, comma separated."),
    "MAP_SHIFT_{b}_ARCSEC": ("arcsec (east, north)",
        "For each row of MAP_ID_{b}: how far the map's own astrometry sits from the SESNA positions, east then north "
        "(map minus SESNA), the median over the map's bright point sources paired one-to-one with a SESNA source within "
        "one beam; 0, 0 when fewer than 10 pairs exist. Every catalogue position is moved by this shift before it is "
        "measured on the map, so a source is measured where the map puts it."),
    "MAP_N_PAIRS_{b}": ("pairs",
        "For each row of MAP_ID_{b}: the number of bright-source pairs MAP_SHIFT_{b}_ARCSEC was measured from (below 10: no shift applied)."),
    "CELL_ID_{b}": ("cell id",
        "For each row of CURVE_{b}: the cell's identifier within its map, row-major from the map's first pixel "
        "row (id = iy x n_cells_x + ix, with ix = floor((x + 0.5) / step_x), iy likewise, x and y the map's own "
        "0-based pixel coordinates, step from MAP_CELL_STEP_{b}; n_cells_x = ceil(MAP_SHAPE x / step_x))."),
    "CELL_MAP_INDEX_{b}": ("row of MAP_ID_{b}",
        "For each row of CURVE_{b}: which map (row of the MAP_* datasets) the cell belongs to."),
    "MAP_WCS_{b}": ("FITS header text",
        "For each row of MAP_ID_{b}: the map's celestial world-coordinate header, as text (astropy "
        "WCS(fits.Header.fromstring(...))). A sky position is moved by MAP_SHIFT_{b}_ARCSEC before this "
        "header turns it into map pixels."),
    "MAP_SHAPE_{b}": ("pixels (ny, nx)",
        "For each row of MAP_ID_{b}: the map's image size, rows then columns."),
    "MAP_CELL_STEP_{b}": ("pixels (x, y)",
        "For each row of MAP_ID_{b}: the cell size in the map's pixels along x and y (30 arcsec over the "
        "pixel scale); edge cells are clipped to the image."),
    "CURVE_PSF_{b}": ("completeness fraction x600, uint16",
        "Same layout as CURVE_{b} (same rows, same MAG_GRID_{b}), but for a source measured by the joint point-source "
        "fit (METHOD 2) instead of the aperture: the probability that a point source of the flux in MAG_GRID_{b} "
        "placed in the cell is detected at 5 sigma or better when measured by that fit, judged against "
        "CELL_OFFSET_PSF_{b}_MJY and CELL_SIGMA_PSF_{b}_MJY."),
    "CELL_OFFSET_PSF_{b}_MJY": ("mJy",
        "For each row of CURVE_{b}: the median flux the joint point-source fit returns at the cell's random empty-sky "
        "positions farther than one beam from every bright source. The fit's counterpart of CELL_OFFSET_{b}_MJY."),
    "CELL_SIGMA_PSF_{b}_MJY": ("mJy",
        "For each row of CURVE_{b}: the one-sigma uncertainty of a {w} micron flux measured by the joint point-source "
        "fit, found from the same empty-sky positions in the same way as CELL_SIGMA_{b}_MJY (tail factor included). "
        "A source measured by the fit is called detected at 5 times this value."),
    "CELL_TAIL_FACTOR_PSF_{b}": ("dimensionless",
        "For each row of CURVE_{b}: the factor by which the joint fit's noise in the cell was raised so that negative "
        "excursions on empty sky are no more frequent than for Gaussian noise; the counterpart of "
        "CELL_TAIL_FACTOR_{b}. CELL_SIGMA_PSF_{b}_MJY already includes it."),
    "CELL_GAIN_PSF_{b}": ("dimensionless",
        "For each row of CURVE_{b}: the median fraction of an injected point source's flux that the joint "
        "point-source fit returns in the cell (1 when it recovers it fully)."),
    "CELL_GAIN_{b}": ("dimensionless",
        "The median fraction of an injected point source's flux that the measurement returns in each "
        "cell of CURVE_{b} (1 when the fit recovers it fully)."),
    "SOURCE_ROW": ("row index into CURVE_<band>",
        "For each curated source, in the order of the region's source table, and the bands P70, P100, P160: "
        "the row of CURVE_P70 / CURVE_P100 / CURVE_P160 for the cell the source falls in on the deepest map covering "
        "it; -1 where no map covers the source in that band. A source near a map edge can have a row "
        "here yet be marked not covered (ORIGIN 91) in the photometry, because its own measurement needs "
        "most of its beam on valid data; the photometry's ORIGIN decides."),
}
for _band, _w in (("P70", "70"), ("P100", "100"), ("P160", "160")):
    for _name, (_u, _r) in _PACS_COMPLETENESS_CELLS_READINGS.items():
        REGISTRY[("completeness_pacs_source", _name.format(b=_band))] = (
            _u, _r.format(b=_band, w=_w, n=200, mmin="-3.5", mmax="5.5", dm="0.25"))

# PACS source photometry (sky.derived.pacs_photometry).
_PHOTOMETRY_PACS_READINGS = {
    "NAME": ("source identifier",
        "The SESNA source identifier of each row, in exactly the order of the region's SESNA source catalogue."),
    "FNU_{b}_MJY": ("mJy",
        "Total flux of the source in the Herschel PACS {w} micron band (the PACS calibration's definition of a "
        "point source's flux), measured with the eHOPS survey's apertures (Pokhrel et al. 2023): aperture "
        "photometry at the catalogue position (moved by the map's astrometric shift, MAP_SHIFT in the completeness "
        "product) on the map that is deepest there, aperture radius 9.6 / 8.0 / 12.8 arcsec at 70 / 100 / 160 "
        "micron, sky from a clipped mode in the annulus 12.8-25.6 / 9.6-16.0 / 16.0-22.4 arcsec, divided by the "
        "fraction of a point source's total flux that this aperture and annulus return for the map's observing "
        "mode and scan speed (PACS encircled-energy tables; 0.71-0.76 at 70, 0.55-0.63 at 100, 0.64-0.65 at 160 "
        "micron; for a parallel-mode map that holds only one scan direction, the fraction is computed from that "
        "direction's own point-spread template, at 70 and 160 micron only: at 100 micron the two templates differ by about 3 "
        "per cent inside 8 arcsec, so such a map is refused there). eHOPS's published fluxes are on a within-60-arcsec normalisation with one constant per band and "
        "sit about 9 % lower at 70 micron, and more at 100 micron on fast-scan maps. Where METHOD is 2 the source has "
        "a neighbour within one beam of a detected source and its flux is the total flux of a joint point-source "
        "fit. It can be negative or small when nothing is seen and is NaN where the band does not cover the source. "
        "Where ORIGIN is 2 it is an upper bound: the combined flux of the blended group plus five times its "
        "uncertainty."),
    "SIGMA_ANNULUS_{b}_MJY": ("mJy",
        "eHOPS's own per-source uncertainty statistic: the scatter of the sky-annulus pixels times the square root of "
        "the aperture area, divided by the aperture correction. Kept for comparison with their tables; detection uses "
        "SIGMA_{b}_MJY."),
    "METHOD_{b}": ("code",
        "1: aperture photometry (the source is isolated); 2: joint point-source fit with its close neighbours; "
        "0: not measured (not covered)."),
    "SIGMA_{b}_MJY": ("mJy, or 0.99 for an upper bound",
        "Uncertainty of the flux: the scatter of the same measurement at random empty-sky positions in the source's "
        "30-arcsecond cell (the median over the 3 x 3 cells around it, raised by the tail factor of the cell's noise "
        "level), enlarged for a source fitted jointly with a neighbour. Where ORIGIN is 2 it holds 0.99, the value the "
        "source table uses to mark an upper bound; NaN where the band does not cover the source. For a jointly fitted "
        "source the uncertainty is enlarged by the overlap but its completeness curve is still its cell's curve, so "
        "the curve somewhat overstates its chance of detection. The noise has heavier tails than a Gaussian; it is "
        "scaled so that 5-sigma excursions on empty sky are as rare as for Gaussian noise, which makes it conservative "
        "in the core (on Musca, empty-sky excursions beyond -3 sigma occur at 0.46 / 0.78 / 0.76 of the Gaussian rate at "
        "70 / 100 / 160 micron)."),
    "ORIGIN_{b}": ("origin code",
        "1 means the source is detected: its flux minus the typical empty-sky flux of its 30-arcsecond cell "
        "(CELL_OFFSET in the completeness product) is at least five times its uncertainty -- the same rule the "
        "completeness curve uses; 90 means the band covers the source but it is not detected; 2 means the source is "
        "blended with a neighbour so closely that the two cannot be separated and the flux is an upper bound for the "
        "pair; 91 means the band does not cover the source."),
    "LOCAL_RMS_{b}_MJY": ("mJy",
        "Noise of the 30-arcsecond cell containing the source: the scatter of the flux measured at random empty-sky positions in that cell. Five times this value is the faintest flux that would be called a detection there. NaN where the band does not cover the source."),
    "OBSID_{b}": ("observation identifier(s)",
        "The Herschel observation identifier(s), comma separated, of the map the flux was measured on; empty where the band does not cover the source."),
    "N_BLEND_{b}": ("number of SESNA sources",
        "How many SESNA sources, including this one, were fitted together because they lie within one beam width of a source detected in this band; 1 for a source fitted alone, 0 where the band does not cover the source."),
    "BLEND_GROUP_{b}": ("row number, or -1",
        "Sources fitted together share this number, which is the row of the first of them in the catalogue; -1 for a source fitted alone or not covered."),
    "EXTENDED_{b}": ("true or false",
        "True for a detected source whose measured width is more than 1.5 times the beam width, a sign that the emission is extended rather than a point source; false otherwise, including for every source that is not detected."),
}
_P160_NOTE = (" At 160 micron: this is the Pokhrel et al. 2023 Table 1 recipe (12.8 arcsec aperture, 16.0-22.4 "
              "arcsec annulus, clipped-mode sky, correction to total flux from the PACS red encircled-energy tables) on "
              "the archive's SPG v14 JScanam maps; the same recipe on the Herschel Gould Belt Survey team's Scanamorphos "
              "maps agrees within 1 per cent, and at 70 micron both agree with eHOPS's published values within 0-2.5 "
              "per cent. At 160 micron eHOPS's published values are 1.22 times what their Table 1 recipe yields on either "
              "map production; the cause cannot be determined from the paper (a wider sky annulus, a different "
              "correction, or a different combination of maps). The band is included in the fit on this convention; the "
              "question to the eHOPS authors is open. The 160 "
              "micron band also contains the [C II] 158 micron line, and its beam is about 12 arcsec.")
for _band, _wave in (("P70", "70"), ("P100", "100"), ("P160", "160")):
    for _name, (_units, _reading) in _PHOTOMETRY_PACS_READINGS.items():
        _text = _reading.format(w=_wave, b=_band)
        if _band == "P160" and _name == "FNU_{b}_MJY":
            _text += _P160_NOTE
        REGISTRY[("photometry_pacs_source", _name.format(b=_band))] = (_units, _text)


# PACS completeness and photometric error measured by recovering artificial sources (2026-10-27,
# PACS_COMPLETENESS_PLAN.md v2). These entries replace the ones above for the datasets whose meaning
# changed, add the four measured-calibration datasets, and drop the two datasets the measurement no
# longer writes (the noise tail factor and the gain).
_PACS_INJECTION_READINGS = {
    "CURVE_{b}": ("completeness fraction x600, uint16",
        "One row per covered 30-arcsec cell of every Herschel PACS {w} micron map, for all cells inside the "
        "region's source bounding box padded by about 7 arcmin (overlapping maps each keep their cells; "
        "SOURCE_ROW points at the deepest one). Each value is the fraction of artificial point sources of the "
        "flux in MAG_GRID_{b} that the catalogue recovers as detections at that cell, stored as the fraction "
        "times 600. The artificial sources are the band's point-spread pattern added to a copy of the real map at "
        "random positions, one per about 50 beam areas at a time, measured exactly as a catalogue source at that "
        "position is (aperture photometry, or the joint fit where a detected source lies within one beam), and "
        "counted as recovered when their flux minus CELL_OFFSET_{b}_MJY reaches 5 times CELL_SIGMA_{b}_MJY, "
        "whatever the flux error. The counts of all cells whose centres lie within 120 arcsec of the cell and of similar noise (CELL_SIGMA within a factor 1.5 of its own) are added "
        "before the fraction is taken, so each value rests on at least 100 injections where the map allows (where fewer result the radius is widened to 240, then 360 arcsec, never the noise window) (N_POS_{b} "
        "gives the number at the faintest plane). The fraction at the faintest planes is the false-positive floor of "
        "the 5-sigma rule as measured, not zero. Rows are ordered by map and then cell."),
    "N_POS_{b}": ("injections",
        "For each row of CURVE_{b}, the number of artificial sources behind its value at the faintest plane of "
        "MAG_GRID_{b}: the injections that fell in the cells within 120 arcsec of the row's cell and of similar noise (CELL_SIGMA within a factor 1.5 of its own) on its map and "
        "lay on valid data, not counting those within half a beam of a detected source. Rows near a map's edge "
        "have fewer, so can fall below the 100 aimed at."),
    "FLUX90_{b}_MJY": ("mJy",
        "For each row of CURVE_{b}, the faintest flux at which the recovered fraction of the artificial sources is "
        "still at least 0.9: the grid read from its bright end, interpolated linearly in the logarithm of the flux "
        "to where the fraction crosses 0.9. NaN where even the brightest plane does not reach 0.9; the faintest "
        "plane's flux if no plane falls below it."),
    "BIAS_{b}_MJY": ("mJy",
        "For each row of CURVE_{b} and each plane of MAG_GRID_{b}: the median of (recovered flux minus injected "
        "flux) over the recovered artificial sources (those counted as detections) measured by aperture photometry "
        "in the cells within 120 arcsec of the row's cell and of similar noise (CELL_SIGMA within a factor 1.5 of its own). The recovered flux is the flux the catalogue reports, "
        "without CELL_OFFSET_{b}_MJY subtracted. NaN where fewer than 100 recovered sources lie in the "
        "neighbourhood at that plane."),
    "SCATTER_{b}_MJY": ("mJy",
        "For each row of CURVE_{b} and each plane of MAG_GRID_{b}: the photometric error of a detected {w} micron "
        "source of that flux at that cell, 1.4826 times the median absolute deviation about the median of "
        "(recovered flux minus injected flux) over the recovered artificial sources measured by aperture "
        "photometry in the cells within 120 arcsec and of similar noise (CELL_SIGMA within a factor 1.5 of its own). This is the uncertainty the photometry product assigns to a "
        "detected, isolated source. NaN where fewer than 100 recovered sources lie in the neighbourhood at that "
        "plane."),
    "SCATTER_BLEND_{b}_MJY": ("mJy",
        "The same as SCATTER_{b}_MJY for a source fitted jointly with a detected neighbour within one beam (the "
        "artificial source is one more member of the fitted group). It comes from a separate set of artificial sources: "
        "for every detected source of the map, 40 per flux step in the ring from half a beam to one beam around it. "
        "1.4826 times the median absolute deviation about the median of (recovered flux minus injected flux) over the "
        "recovered ones in the cells within 120 arcsec (240 or 360 where too few) of the row's cell and of similar noise. "
        "NaN where fewer than 30 recovered ones are in reach, which is most cells: only cells near a detected source "
        "have a value, and the photometry uses the nearest such cell."),
    "CELL_SIGMA_{b}_MJY": ("mJy",
        "The noise of a {w} micron flux measured in each cell of CURVE_{b}, used only to decide detection: the "
        "median, over the 3 x 3 block of cells around it, of 1.4826 times the median absolute deviation about "
        "CELL_OFFSET_{b}_MJY of the fluxes measured at each cell's random positions farther than one beam from "
        "every bright source (one cell alone holds only about ten independent beams). A flux is a detection when "
        "it minus CELL_OFFSET_{b}_MJY is at least 5 times this value. It is not the uncertainty of a detected "
        "source (that is SCATTER_{b}_MJY) and no tail correction is applied."),
}
for _band, _w in (("P70", "70"), ("P100", "100"), ("P160", "160")):
    for _name, (_u, _r) in _PACS_INJECTION_READINGS.items():
        REGISTRY[("completeness_pacs_source", _name.format(b=_band))] = (_u, _r.format(b=_band, w=_w))
    for _gone in ("CELL_TAIL_FACTOR_{b}", "CELL_GAIN_{b}"):
        REGISTRY.pop(("completeness_pacs_source", _gone.format(b=_band)), None)
    REGISTRY[("photometry_pacs_source", "SIGMA_%s_MJY" % _band)] = (
        "mJy, or 0.99 for an upper bound",
        "Uncertainty of the flux. For a detected source (ORIGIN 1) it is the photometric error measured by "
        "recovering artificial sources of known flux injected into the map: the scatter of (recovered minus injected "
        "flux) at the source's own flux, in its 30-arcsecond cell's neighbourhood, for the aperture measurement "
        "(METHOD 1) or for the joint fit with a close neighbour (METHOD 2), interpolated in the logarithm of the "
        "flux between the two flux steps that bracket the source's flux minus its cell's empty-sky offset "
        "(SCATTER and SCATTER_BLEND of the completeness product; for a jointly fitted source whose cell has no SCATTER_BLEND, "
        "that of the nearest cell within 360 arcsec on the same map with similar noise, else the aperture SCATTER of its "
        "own cell). For a source that is covered but not detected "
        "(ORIGIN 90) it is the noise of its cell, LOCAL_RMS. Where ORIGIN is 2 it holds 0.99, the value the source "
        "table uses to mark an upper bound; NaN where the band does not cover the source, and where the "
        "completeness product has no measurement for the cell at any flux.")



# Unit 0: PACS bands
# Nothing is registered here. The datasets that carry the three PACS bands
# (curated source table, completeness product, the four libraries' P70/P100/P160
# columns, the PACS far-infrared census) are registered by the units that build
# them, each in its own block; their UNITS and READING are those units'.


# Unit 2: catalog.curated, the eleven-band source table (replaces the eight-band readings above)
_PACS_TAIL = (" The last three columns are the Herschel PACS bands P70, P100, P160, taken from the PACS "
              "photometry file of the same region (see its readings for FNU_P70_MJY etc.).")
REGISTRY[("sources_sesna_source", "FNU_MJY")] = ("mJy",
    "The flux in each of the eleven bands, in the band order this file's BANDS attribute gives (J, H, Ks, I1, I2, "
    "I3, I4, M1, P70, P100, P160). Where there is no detection in the first eight the value is filled: the 2MASS "
    "bands (J, H, Ks) take the survey's fixed detection bound, and the three IRAC bands and the one MIPS band take "
    "the source's own 90%-completeness flux read off Rob Gutermuth's measured completeness map at the source's "
    "nearest pixel, or NaN where the source's own pixel is off every field's map. The three PACS columns are total "
    "flux (the PACS calibration's definition), as measured by the PACS stages; an upper bound or an uncovered band "
    "is as that file records it. ORIGIN_FNU in this file says which bands were filled and how." + _PACS_TAIL)
REGISTRY[("sources_sesna_source", "SIGMA_FNU_MJY")] = ("mJy",
    "The 1-sigma flux uncertainty for a measured detection in FNU_MJY, in each of the eleven bands, on the same mJy "
    "scale. Where FNU_MJY was filled instead of measured (see ORIGIN_FNU), or is a PACS upper bound, this is the "
    "fixed value 0.99 (a flag the SED fitter reads as an upper limit, not a real measurement uncertainty), or NaN "
    "where ORIGIN_FNU is 91 (off-map or not covered, no fill at all)." + _PACS_TAIL)
REGISTRY[("sources_sesna_source", "DCOMP90_MJY")] = ("mJy",
    "The flux a source at this position and band would need to be detected at 90% completeness, in each of the "
    "eleven bands: the survey's fixed bound for the 2MASS bands (J, H, Ks), or, for the three IRAC bands and the "
    "one MIPS band, Rob Gutermuth's measured completeness map value at the source's nearest pixel, NaN where the "
    "source's own pixel is off every field's map for that band. NaN in the three PACS columns: the PACS 90% flux "
    "is in the PACS completeness product, not here.")
REGISTRY[("sources_sesna_source", "ORIGIN_FNU")] = ("code",
    "How FNU_MJY and DCOMP90_MJY in each of the eleven bands were obtained, one integer per band: 1 a measured "
    "detection, 2 the fixed 2MASS survey detection bound (for the PACS bands: an upper bound from a blend that "
    "cannot be separated), 90 covered but undetected (FNU_MJY filled from the completeness map's own "
    "90%-completeness flux at the source's pixel; for the PACS bands: observed and undetected), 91 off-map or not "
    "covered -- no completeness value and no non-detection term (COMPLETENESS_2026-10-20.md rule 2, the same "
    "meaning Pokhrel et al. 2023, eHOPS I, ApJS 266, 32, gave the old catalogue column)." + _PACS_TAIL)
# Unit 3: completeness_sesna_source grows to eleven bands (P70, P100, P160 copied from the PACS input)
for _band, _w in (("P70", "70"), ("P100", "100"), ("P160", "160")):
    for _name, (_u, _r) in _PACS_COMPLETENESS_CELLS_READINGS.items():
        if _name == "SOURCE_ROW":
            continue
        REGISTRY[("completeness_sesna_source", _name.format(b=_band))] = (
            _u, _r.format(b=_band, w=_w, n=200, mmin="-3.5", mmax="5.5", dm="0.25"))
    REGISTRY[("completeness_sesna_source", "METHOD_%s" % _band)] = (
        "code",
        "For each curated source, in the order of the region's source table: how its %s micron flux was measured, "
        "1 aperture photometry (the source is isolated), 2 a joint point-source fit with its close neighbours, "
        "0 not measured (no %s micron map covers it). A source with 2 has its %s micron completeness taken from "
        "CURVE_PSF_%s, any other from CURVE_%s." % (_w, _w, _w, _band, _band))
    REGISTRY[("completeness_sesna_source", "PIXEL_CURVE_%s" % _band)] = (
        "dimensionless (completeness fraction, 0..1)",
        "For each admitted nside-512 pixel in HPX_PIX_512, the %s micron completeness curve at "
        "MAG_GRID_%s's magnitudes: the mean of CURVE_%s over the cells that the pixel's 256 sample points fall "
        "in (each point on the deepest %s micron map that holds a cell there, positions moved by the map's "
        "registration shift), as a fraction (not times 600). Averaged over the covered points only, so it is "
        "the curve where the band has data; all zero where the pixel has no %s micron coverage "
        "(PIXEL_COVERED_FRAC_%s is 0)." % (_w, _band, _band, _w, _w, _band))
    REGISTRY[("completeness_sesna_source", "PIXEL_COVERED_FRAC_%s" % _band)] = (
        "dimensionless (area fraction, 0..1)",
        "For each admitted nside-512 pixel in HPX_PIX_512, the fraction of its 256 sample points that fall in a "
        "stored %s micron cell. A PACS band's coverage is this number; PIXEL_UNION_FRAC concerns the IRAC "
        "bands only and PACS bands are not part of that union." % _w)
REGISTRY[("completeness_sesna_source", "SOURCE_ROW")] = (
    "row index into CURVE_<band>",
    "For every curated source and each of the eleven bands, definitions.BANDS' own order (J, H, Ks, I1, I2, I3, "
    "I4, M1, P70, P100, P160 -- the same order fittp.sweep's BAND_KEYS uses): the row of that band's own "
    "CURVE_<band> holding this source's completeness curve. For I1-M1, its nearest on-map 30-arcsec pixel, -1 "
    "where the source is off every field's map in that band. For J/H/Ks, the row of the nside-128 pixel the "
    "source falls in, never -1 (2MASS is all-sky). For P70/P100/P160, the row of the 30-arcsec cell the source "
    "falls in on the deepest PACS map covering it, copied from the PACS completeness file; -1 where no map "
    "covers the source in that band. A source near a map edge can have a row here yet be marked not covered "
    "(ORIGIN 91) in the PACS photometry; the photometry's ORIGIN decides.")
