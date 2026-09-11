"""The words the two prior pages share: the vocabulary of the prior's
quantities and the probability each panel draws, defined once here and
printed on both pages (`atlas.shapes`, `atlas.render`), so a reader never
has to guess what a term means or which probability a panel shows.
"""

#: The vocabulary, in the order a reader meets the terms. Each entry is
#: (term, definition); the pages print them as one block.
VOCABULARY = (
    ("source position s",
     "a catalogued source's sky position, with its own sightline column A_s "
     "(the extinction through the whole sightline, in A_K magnitudes) and its "
     "own detection limits"),
    ("scaled extinction x",
     "a / A_s, the extinction in front of an object as a fraction of its "
     "sightline's column; x <= 1 by definition"),
    ("F_4.5",
     "the object's 4.5 micron flux density, in mJy"),
    ("parameter cell",
     "one bin of the prior's common grid over (log10 x, log10 F_4.5): "
     "1/32 dex in log10 x by 0.1 dex in log10 F_4.5; every cell has the same area"),
    ("sky pixel",
     "one HEALPix nside-512 pixel of the atlas (6.9 arcmin on a side)"),
    ("class C",
     "one of STAR, AGB, PAHC, YSO, H2S, GAL"),
    ("intensity A_C(s)",
     "the prior's sky density of class C at s, per square degree, before the "
     "survey's selection"),
    ("shape h_C(x, F_4.5)",
     "the prior's density of class C over the parameter plane at s, of unit "
     "mass, per dex^2"),
    ("weight factor f_C(F_4.5; s)",
     "the template-weight factor of class C at that flux (1 for a class whose "
     "factors are normalised over its templates)"),
    ("prior density Lambda_C",
     "Lambda_C(x, F_4.5; s) = A_C(s) h_C(x, F_4.5) f_C(F_4.5; s), the density "
     "the fitter compares between classes"),
    ("catalogued",
     "passing the survey's selection at the sky pixel: two of the eight bands "
     "measured above the pixel's limits"),
)

#: The probability each panel draws, one statement per panel kind.
SHAPES_ROW1 = (
    "Row 1, per class: P(C, cell | s) = Lambda_C(cell) / sum over classes and "
    "cells of Lambda(cell) -- the prior probability that a source at s is of "
    "class C AND lies in this parameter cell. One log colour scale for all six "
    "panels, so the panels compare cell by cell.")
SHAPES_ROW2 = (
    "Row 2, per class: P(C | cell, s) = Lambda_C(cell) / sum over classes of "
    "Lambda(cell) -- the prior probability that a source at s known to lie in "
    "this parameter cell is of class C. Where every class is at the common "
    "floor the six shares are equal: the prior places no source there.")
ATLAS_INTRINSIC_CLASS = (
    "Class panels: P(C | sky pixel) = A_C(pixel) / sum over classes of "
    "A(pixel) -- the prior probability that a source in this sky pixel is of "
    "class C, before the survey's selection, drawn on a log scale from 10^-4 "
    "to 1.")
ATLAS_INTRINSIC_TOTAL = (
    "Density panel: sum over classes of A_C(pixel), per square degree -- the "
    "prior's sky density of sources, before the survey's selection.")
ATLAS_SELECTION_CLASS = (
    "Class panels: P(C | sky pixel, catalogued) = N_C(pixel) / sum over classes "
    "of N(pixel), with N_C the prior's density of catalogued sources of class C "
    "-- the prior probability that a catalogued source in this sky pixel is of "
    "class C, drawn on a log scale from 10^-4 to 1.")
ATLAS_SELECTION_TOTAL = (
    "Density panel: sum over classes of N_C(pixel), per square degree -- the "
    "prior's sky density of catalogued sources.")


def vocabulary_block():
    """The vocabulary as one string, one term per line, for a page's caption."""
    return "\n".join("%s: %s" % (term, definition) for term, definition in VOCABULARY)
