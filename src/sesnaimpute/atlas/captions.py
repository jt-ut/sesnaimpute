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

#: The total-count check (sec. 8, sec. 9), the selection page's caption
#: line: `{value}` is the region's own N_prior / N_catalog ratio.
TOTAL_COUNT_CHECK = (
    r"$N_{{prior}} / N_{{catalog}}$ = {value:.3f}: the prior's expected number "
    "of cataloged sources in the region (the selection densities summed over "
    "the sky pixels, times the surveyed area) against the number of sources "
    "in the catalog -- the check that the prior's normalization reproduces "
    "the survey's count; the observations enter here only as that check.")


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
