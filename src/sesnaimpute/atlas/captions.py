"""The words the two prior pages share: the vocabulary of the prior's
quantities and the probability each panel draws, defined once here and
printed on both pages (`atlas.shapes`, `atlas.render`), so a reader never
has to guess what a term means or which probability a panel shows.
"""

import textwrap

#: The vocabulary, in the order a reader meets the terms. Each entry is
#: (term, definition); the pages print them as one block.
VOCABULARY = (
    ("source position s",
     "a cataloged source's sky position, with its own sightline column A_s "
     "(the extinction through the whole sightline, in A_K magnitudes) and its "
     "own detection limits"),
    ("scaled extinction x",
     "a / A_s, the extinction in front of an object as a fraction of its "
     "sightline's column; x <= 1 by definition"),
    ("$F_{4.5}$",
     "the object's 4.5 micron flux density, in mJy"),
    ("parameter cell",
     "one bin of the prior's common grid over (log10 x, log10 $F_{4.5}$): "
     "1/32 dex in log10 x by 0.1 dex in log10 $F_{4.5}$; every cell has the same area"),
    ("sky pixel",
     "one HEALPix nside-512 pixel of the atlas (6.9 arcmin on a side)"),
    ("IRAC footprint",
     "the fraction of a sky pixel's area the IRAC catalog covers (COVERAGE: "
     "the fraction of its sixteen nside-2048 sub-pixels holding a cataloged "
     "source with an IRAC detection)"),
    ("class C",
     "one of STAR, AGB, PAHC, YSO, H2S, GAL"),
    ("intensity A_C(s)",
     "the prior's sky density of class C at s, per square degree, before the "
     "survey's selection"),
    ("shape h_C(x, $F_{4.5}$)",
     "the prior's density of class C over the parameter plane at s, of unit "
     "mass, per unit log10 x per unit log10 $F_{4.5}$, that is per dex^2 (an "
     "interval of log10 x is a dex whether or not x carries a unit)"),
    ("weight factor f_C($F_{4.5}$; s)",
     "the template-weight factor of class C at that flux (1 for a class whose "
     "factors are normalized over its templates)"),
    ("prior density Lambda_C",
     "Lambda_C(x, $F_{4.5}$; s) = A_C(s) h_C(x, $F_{4.5}$) f_C($F_{4.5}$; s), "
     "the density the fitter compares between classes"),
    ("cataloged",
     "passing the survey's selection at the sky pixel: two of the eight bands "
     "measured above the pixel's limits"),
    ("measured coordinate r = a_hat / A_beam",
     "the fitted foreground over the sightline's own BEAM-averaged column, rather than "
     "x's own true pencil column; the column kernel carries a class's shape past the wall "
     "(log10 r > 0, the array's one-dex padding) into real mass there -- a pencil column "
     "above the beam mean, never folded back -- so the source page draws it past the x = 1 line; "
     "the region page's grids are in x itself and end at the wall"),
)

#: The probability each panel draws, one statement per panel kind.
SHAPES_ROW1 = (
    "Row 1, per class: P(C, cell | s) = Lambda_C(cell) / sum over classes and "
    "cells of Lambda(cell) -- the prior probability that a source at s is of "
    "class C AND lies in the parameter cell at (x, F_4.5). One log color "
    "scale for all six panels, so the panels compare cell by cell.")
SHAPES_ROW2 = (
    "Row 2, per class: P(C | cell, s) = Lambda_C(cell) / sum over classes of "
    "Lambda(cell) -- the prior probability that a source at s known to lie in "
    "the parameter cell at (x, F_4.5) is of class C. Where every class is at "
    "the common floor the six shares are equal: the prior places no source "
    "there.")
ATLAS_INTRINSIC_CLASS = (
    "Class panels: P(C | sky pixel, brighter at 4.5 micron than the sky "
    "pixel's own 50% completeness limit (the depth grid, from the sources' "
    "own IRAC limits)) = N_ABOVE_C(pixel) / sum over classes of "
    "N_ABOVE(pixel) -- the prior probability that a source in this sky "
    "pixel, above that limit, is of class C.")
ATLAS_INTRINSIC_TOTAL = (
    "Density panel: sum over classes of N_ABOVE_C(pixel), per square degree "
    "-- the prior's sky density of sources brighter at 4.5 micron than the "
    "sky pixel's own 50% completeness limit (the depth grid, from the "
    "sources' own IRAC limits).")
ATLAS_SELECTION_CLASS = (
    "Class panels: P(C | sky pixel, cataloged) = N_C(pixel) / sum over classes "
    "of N(pixel), with N_C the prior's density of cataloged sources of class C "
    "-- the prior probability that a cataloged source in this sky pixel is of "
    "class C.")
ATLAS_SELECTION_TOTAL = (
    "Density panel: sum over classes of N_C(pixel), per square degree -- the "
    "prior's sky density of cataloged sources; the white outline marks the "
    "IRAC footprint at one half.")

#: The region shapes page's three rows (`atlas.shapes.build_region_prior`,
#: SPEC_BMSTP_DRAFT.md sec. 8): P6's own per-cell region grids, read and
#: never recomputed (rule 5).
SHAPES_REGION_ROW_TITLES = (
    "Row 1 -- N_CAT_CELL: the cataloged count per cell",
    "Row 2 -- the class share among cataloged objects",
    "Row 3 -- N_CELL: the intrinsic prior density (selection-free)",
)
SHAPES_REGION_ROW1 = (
    "Row 1, per class: N_CAT_CELL_C(cell) -- the region's expected number "
    "of cataloged objects of class C per parameter cell (bmstp.atlas P6, "
    "sec. 8), summing over cells to RATIO_C * TOTAL_OBSERVED. One log "
    "color scale for all six panels, so the panels compare cell by cell; "
    "a cell below one millionth of the row's peak is white, and each panel's "
    "title carries that class's own peak, so a class whose whole count sits "
    "below that floor draws white by that rule and not for want of mass.")
SHAPES_REGION_ROW2 = (
    "Row 2, per class: N_CAT_CELL_C(cell) / sum over classes of "
    "N_CAT_CELL(cell) -- the class share of the region's cataloged "
    "objects at that cell, drawn wherever the denominator is nonzero and "
    "white elsewhere. The two contours over each panel enclose 50% and "
    "99% of that class's own cataloged mass (row 1's own density), so "
    "the eye reads the share where the population lives.")
SHAPES_REGION_ROW3 = (
    "Row 3, per class: N_CELL_C(cell) -- the region's UNTHINNED intrinsic "
    "population of class C per parameter cell (bmstp.atlas P6, sec. 8), "
    "before the survey's selection: no flux cut, no dimming. One log "
    "color scale for all six panels, the same floor and the same peak-in-title "
    "convention as row 1.")
#: The region totals line, one per class (rule 7a): `{n_cat}` = sum
#: N_CAT_CELL_C = the class's cataloged count, `{n_cell}` = sum N_CELL_C =
#: its intrinsic count, `{ratio}` their ratio = the class's catalogable
#: fraction.
SHAPES_REGION_TOTALS = (
    "{cls}: cataloged={n_cat:.6g}, intrinsic={n_cell:.6g}, "
    "catalogable fraction={ratio:.6f}")

#: The total-count check (sec. 8, sec. 9), the selection page's caption
#: line: `{value}` is the region's own N_prior / N_catalog ratio.
TOTAL_COUNT_CHECK = (
    r"$N_{{prior}} / N_{{catalog}}$ = {value:.3f}: the prior's expected number "
    "of cataloged sources in the region (the selection densities summed over "
    "the sky pixels, times the surveyed area) against the number of sources "
    "in the catalog -- the check that the prior's normalization reproduces "
    "the survey's count; the observations enter here only as that check.")

#: The protostar check's own words (`atlas.protostars`, SPEC_BMSTP_DRAFT.md
#: sec. 5.5 "Check (report only)", sec. 9's protostar row), rule 7a.
PROTOSTAR_STATEMENT = (
    "For each Herschel-confirmed protostar, assumed a SESNA source with its "
    "own 4.5 micron datum: at its own point of the nuisance plane (scaled "
    "extinction x, its dereddened $F_{4.5}$), P(C | x, datum, s) reads the "
    "prior density Lambda_C at that point against the one likelihood factor "
    "the datum implies -- a Gaussian in log10 F for a measurement, the "
    "survey's own non-detection probability below the pixel's limit "
    "otherwise -- normalized over the six classes; nothing here feeds the "
    "prior or a fit.")

PROTOSTAR_POSITION = (
    "position: at the protostars' own sky pixels the prior's cataloged YSO "
    "share has median {median_proto:.3f}, against {median_source:.3f} at "
    "every cataloged source's own pixel.")

PROTOSTAR_DEPTH = (
    "depth: {n_past_wall} of {n_used} protostars' own foreground column exceeds "
    "their sightline's own BEAM column $A_\\mathrm{{beam}}$ (the extinction column of "
    "sec. 3.2's own A_COL_K) and read past the wall in the measured coordinate "
    "(log10 r_p > 0) -- a beam column is the average along the whole line of sight, "
    "while a protostar's own fitted foreground reads its own envelope, so a protostar "
    "can sit past the wall by construction, not by a match error; {n_beyond_reach} of "
    "{n_used} lie beyond the kernel's reach (P(T >= a_p) < 0.01) and are drawn at the "
    "wall with an open marker and no interval.")

PROTOSTAR_KERNEL_CHECK = (
    "kernel check: this compares the column kernel's own 36 arcsec-beam width, "
    "extrapolated to a cloud member's own beam ratio log10 r_p = log10 x_member + y "
    "(x_member from the region's median-sightline YSO x marginal, y from each "
    "protostar's own class kernel at its column), against the sample's own ratio "
    "distribution -- empirical log10 r_p has median {emp_median:.3f}, 84th percentile "
    "{emp_p84:.3f} over {n_used} protostars; the kernel predicts median {pred_median:.3f}, "
    "84th percentile {pred_p84:.3f}; {frac_beyond:.3f} of protostars lie beyond the "
    "kernel's reach.")

PROTOSTAR_TIERS = (
    "of {n_used} protostars used: {n_measured} with a measured 4.5 micron "
    "flux, {n_not_measured} without one; {n_direct} matched a cataloged "
    "source within 2 arcsec, {n_standin} took the nearest cataloged source "
    "in their own sky pixel as a sightline stand-in, {n_excluded} had no "
    "cataloged source in their own pixel and are excluded.")


def vocabulary_block():
    """The vocabulary as one string, one term per line, for a page's caption."""
    return "\n".join("%s: %s" % (term, definition) for term, definition in VOCABULARY)


def caption_layout(text, wrap_chars, line_height_in, top_pad_in, bottom_pad_in):
    """`(wrapped_text, height_in)`: `text` hard-wrapped at `wrap_chars`
    characters per line (matplotlib draws `fig.text` as given, wrapping
    nothing on its own), and the height, inches, a page must grow by to
    hold exactly that wrapped text at `line_height_in` per line plus
    `top_pad_in`/`bottom_pad_in` -- the one definition both prior pages'
    layouts use (`atlas.render`, `atlas.shapes`), so the drawn block and
    the reserved strip always agree."""
    wrapped = "\n".join("\n".join(textwrap.wrap(line, wrap_chars)) if line else ""
                         for line in text.split("\n"))
    n_lines = wrapped.count("\n") + 1
    height = top_pad_in + n_lines * line_height_in + bottom_pad_in
    return wrapped, height
