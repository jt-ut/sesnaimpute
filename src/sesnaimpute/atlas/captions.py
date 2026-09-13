"""The words the two prior pages share: the vocabulary of the prior's
quantities and the probability each panel draws, defined once here and
printed on both pages (`atlas.shapes`, `atlas.render`), so a reader never
has to guess what a term means or which probability a panel shows.
"""

import os
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
    ("selected",
     "passing the survey's selection at the sky pixel: two of the eight bands "
     "measured above the pixel's limits (the catalog's own condition; also "
     "called cataloged)"),
    ("measured coordinate r = a_hat / A_beam",
     "the fitted foreground over the sightline's own BEAM-averaged column, rather than "
     "x's own true pencil column; the column kernel carries a class's shape past the wall "
     "(log10 r > 0, the array's one-dex padding) into real mass there -- a pencil column "
     "above the beam mean, never folded back -- so the source page draws it past the x = 1 line; "
     "the region page's grids are in x itself and end at the wall"),
)

#: The probability each panel draws, one statement per panel kind.
SHAPES_ROW1 = (
    "Top panel: P($\\hat{\\xi}$, F_4.5 | s) = Σ_C Λ_C(cell) / Σ_C Σ_cells "
    "Λ_C(cell) -- the prior probability that a source at s lies in the cell, summed "
    "over the six classes: where the prior expects a source at this position.")
SHAPES_ROW2 = (
    "Row 2, per class: P(C | cell, s) = Lambda_C(cell) / sum over classes of "
    "Lambda(cell) -- the prior probability that a source at s known to lie in "
    "the parameter cell at ($\\hat{\\xi}$, F_4.5) is of class C. Where every class is at "
    "the common floor the six shares are equal: the prior places no source "
    "there. Each cell is drawn at opacity 1 − H / ln 6, H the entropy of the six "
    "shares: solid where the prior decides the class, faded to white where it is "
    "blind (every class at the common floor, the six shares equal).")
SHAPES_MEASURED = (
    "$\\hat{\\xi}$ = â_s / A_s, the fitted foreground extinction as a fraction of the "
    "sightline's column; the shapes are drawn as the fitter reads them, blurred by "
    "the measured ratio's scatter, which is why mass can lie past $\\hat{\\xi}$ = 1.")
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
    "Class panels: P(C | sky pixel, selected) = N_C(pixel) / sum over classes "
    "of N(pixel), with N_C the prior's density of selected sources of class C "
    "-- the prior probability that a selected source in this sky pixel is of "
    "class C.")
ATLAS_SELECTION_TOTAL = (
    "Density panel: sum over classes of N_C(pixel), per square degree -- the "
    "prior's sky density of selected sources; the white outline marks the "
    "IRAC footprint at one half.")

#: The region shapes page's one row (`atlas.shapes.build_region_prior`,
#: SPEC_BMSTP_DRAFT.md sec. 8): P6's own per-cell region grid, read and
#: never recomputed (rule 5).
SHAPES_REGION_ROW1 = (
    "Each panel: N_C(cell), the number of selected objects of class C the "
    "prior expects in the region per parameter cell (ξ, F_4.5), summed "
    "over the region's sources; one log colour scale for the six panels, "
    "white at one millionth of the row's peak. The dotted line is the "
    "region's median 4.5 µm 50 % completeness limit; the dashed line "
    "is ξ = 1.")
#: The region totals line, one per class (rule 7a): `{n_cat}` = sum
#: N_CAT_CELL_C = the class's selected count, `{n_cell}` = sum N_CELL_C =
#: its count before selection, `{ratio}` their ratio = the class's
#: catalogable fraction.
SHAPES_REGION_TOTALS = (
    "{cls}: selected {n_cat:.6g} of {n_cell:.6g} before selection "
    "(fraction {ratio:.3f})")

#: The total-count check (sec. 8, sec. 9), the selection page's caption
#: line: `{value}` is the region's own N_prior / N_catalog ratio.
TOTAL_COUNT_CHECK = (
    r"$N_{{prior}} / N_{{catalog}}$ = {value:.3f}: the prior's expected number "
    "of selected sources in the region (the selection densities summed over "
    "the sky pixels, times the surveyed area) against the number of sources "
    "in the catalog -- the check that the prior's normalization reproduces "
    "the survey's count; the observations enter here only as that check.")

#: The protostar check's own words (`atlas.protostars`, SPEC_BMSTP_DRAFT.md
#: sec. 5.5 "Check (report only)", sec. 9's protostar row; rewritten to the
#: owner's 2026-09-13 ruling, plain words only, four short parts).
PROTOSTAR_PLANE = (
    r"Each protostar at its measured depth fraction $\hat{\xi}$ = â / A_s "
    "(its fitted foreground extinction over its sightline's column) and its "
    "dereddened 4.5 µm flux, coloured by the prior's probability that a "
    "source with that flux at that position is a young star.")

PROTOSTAR_VERDICT = (
    "The prior favours a young star for {n_lead} of {n}; median P(young "
    "star) = {median:.3f}.")

#: `{far_clause}` is `; {n_far} of these far above it.` when `n_far > 0`,
#: else `.` (the sentence's own close) -- built by the caller, never a
#: second template, so the clause only appears when there is one to name.
PROTOSTAR_CASES = (
    r"{n_past} of {n} have a fitted foreground extinction above their "
    r"sightline's column (past $\hat{{\xi}}$ = 1){far_clause} {n_nofg} have "
    "no fitted foreground and take the verdict averaged over depth. "
    "{n_noflux} have no 4.5 µm flux and are drawn at their detection "
    "limit.")

PROTOSTAR_RATIO = (
    "Protostars per young star the prior expects in the footprint: "
    "{ratio:.3f} (Dunham et al. 2014: {dunham:g}).")


def vocabulary_block():
    """The vocabulary as one string, one term per line, for a page's caption."""
    return "\n".join("%s: %s" % (term, definition) for term, definition in VOCABULARY)


def write_vocabulary(out_dir):
    """Writes `vocabulary.txt` under `out_dir`: a one-line header naming
    what the file is, then `vocabulary_block()` in full. The prior
    pages' own captions carry the two panel statements (and, on the
    selection page, the total-count check) but not this block, so a
    caller writes it once per figure built; an idempotent overwrite,
    since every call writes the same content to the same path."""
    path = os.path.join(out_dir, "vocabulary.txt")
    with open(path, "w") as f:
        f.write("Vocabulary for the prior atlas pages (prior-atlas-intrinsic/"
                 "prior-atlas-selection):\n\n")
        f.write(vocabulary_block())
        f.write("\n")
    return path


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
