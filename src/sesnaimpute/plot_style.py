"""Package-wide plotting style for sesnacomplete.

Any figure produced anywhere in this package should call `apply_style()`
once (idempotent -- plain rcParams assignment, safe to call repeatedly)
before creating plots, and use `FIGURE_SIZES`/`new_figure`/`savefig`
below rather than picking a size or output format ad hoc per script --
the point of this module is that a plot's origin (which script, which
subpackage) shouldn't be visible in how it looks.

First-draft scope (figure sizing, output format, font, bold titles) --
more will be added here as the package's plotting needs grow. Add to
this file rather than starting a second, parallel styling convention
elsewhere.
"""

import os

import matplotlib.pyplot as plt

# Standard figure sizes (inches), by orientation. Pick the one matching
# the plot's natural aspect rather than inventing a one-off size per
# script -- e.g. a wide time series or bar chart is "landscape", a
# scatter/heatmap with comparable axis ranges is "square", a tall
# single-column figure is "portrait".
#
# "two_panel" is deliberately exactly two "landscape" panels side by
# side (2 x 6.0 wide, same 4.0 height), not a new aspect: a multi-panel
# figure should read as the single-panel figures repeated, and deriving
# it from an existing entry is what keeps that true if the base sizes
# are ever retuned. Verified adequate for a two-panel figure carrying a
# colorbar and per-panel titles -- nothing clips under tight_layout.
# Use it with `new_figure(kind="two_panel", ncols=2)`.
FIGURE_SIZES = {
    "landscape": (6.0, 4.0),
    "square": (6.0, 6.0),
    "portrait": (4.0, 6.0),
    "two_panel": (12.0, 4.0),
}


# The house convention for units on a label: the unit goes in square
# brackets after the quantity. Stated once, here, so that changing it
# repaints every figure that composes its labels through `label()`
# instead of baking the punctuation into each stored string.
UNIT_FORMAT = "{quantity} [{unit}]"


def label(quantity, unit=None):
    """`quantity` with `unit` appended per `UNIT_FORMAT`, or `quantity`
    unchanged when `unit` is None (a dimensionless or unitless
    quantity). Use this rather than writing the brackets by hand, so
    the convention has exactly one definition."""
    if unit is None:
        return quantity
    return UNIT_FORMAT.format(quantity=quantity, unit=unit)


def apply_style():
    """Set the package-wide matplotlib rcParams (output format, font,
    bold titles/axis-labels). Call once per script/session before
    creating any figure."""
    plt.rcParams.update({
        "savefig.format": "pdf",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "figure.titleweight": "bold",   # fig.suptitle(...)
        "axes.titleweight": "bold",     # ax.set_title(...)
        "axes.labelweight": "bold",     # ax.set_xlabel/set_ylabel(...)
    })


def new_figure(kind="landscape", **kwargs):
    """New (fig, ax) at the package's standard size for `kind`
    (see FIGURE_SIZES). Does not call `apply_style()` itself -- that's a
    once-per-script call, not once-per-figure. Extra `kwargs` are passed
    through to `plt.subplots` (e.g. `nrows`/`ncols`).

    For a side-by-side pair use `new_figure(kind="two_panel", ncols=2)`
    rather than bypassing this helper to pass a `figsize` -- the size
    belongs in FIGURE_SIZES so every multi-panel figure in the package
    is the same shape. A caller-supplied `figsize` is rejected for the
    same reason: it is how per-script one-off sizes get back in, which
    is exactly what this module exists to prevent. If no entry fits,
    add one here."""
    if kind not in FIGURE_SIZES:
        raise ValueError(f"unknown figure kind {kind!r}; expected one of {list(FIGURE_SIZES)}")
    if "figsize" in kwargs:
        raise TypeError(
            "new_figure() does not accept figsize -- pick a `kind` from "
            f"{list(FIGURE_SIZES)}, or add a new entry to FIGURE_SIZES. "
            "Previously this raised an opaque 'multiple values for keyword "
            "argument figsize' from plt.subplots.")
    return plt.subplots(figsize=FIGURE_SIZES[kind], **kwargs)


def style_legend(legend):
    """Bold a legend's title. matplotlib has no rcParam for legend-title
    font weight (only `legend.title_fontsize`), so this must be applied
    per-legend, after creation -- e.g. `style_legend(ax.legend(title=...))`.
    No-op if the legend has no title set. Returns `legend` for chaining."""
    title = legend.get_title()
    if title is not None and title.get_text():
        title.set_fontweight("bold")
    return legend


def savefig(fig, path, **kwargs):
    """Save `fig` as a PDF (package convention -- see `apply_style`'s
    `savefig.format`) to `path`, creating parent directories if needed.
    `path` may omit the `.pdf` extension; it's added if missing, never
    appended twice. Returns the final path written."""
    path = str(path)
    if not path.lower().endswith(".pdf"):
        path += ".pdf"
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, **kwargs)
    return path


def new_sized_figure(width_in, height_in):
    """A bare figure at an explicit size, for the one case `FIGURE_SIZES`
    cannot serve: a figure whose page size is a FUNCTION OF THE DATA,
    not a choice -- an atlas whose panels each carry their region's own
    sky-footprint aspect ratio, where a fixed page would letterbox one
    region and crush the next (the built regions' footprint aspects span
    0.4 to 5.0).

    This is the narrow exception, not a way back to per-script sizes:
    the caller must be deriving both numbers from the data. Anything
    whose size is a preference belongs in `FIGURE_SIZES` and goes
    through `new_figure`. Returns the figure only -- a caller in this
    situation is placing its own axes.
    """
    return plt.figure(figsize=(width_in, height_in))
