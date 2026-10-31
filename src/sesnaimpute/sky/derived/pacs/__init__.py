"""The PACS build, consolidated (FIX_SPEC.md sec 11; the contract: bms_review/briefs/pacs_consolidation_2026-10-31/
CONTRACT.md). Modules in dependency order -- each imports only modules before it:

    grid <- convert <- psf <- mosaic <- detect <- inject <- photometry <- products <- checks <- stages <- driver

Round 1 (2026-10-31) wrote `grid` in full and every other module as signatures; the old modules
sky/derived/pacs_*.py stay the working build until round 3 switches the callers over.

The submodules load on first attribute access (`sesnaimpute.sky.derived.pacs.grid`), so importing the package costs
nothing and imports no module out of order.
"""

import importlib

MODULES = ("grid", "convert", "psf", "mosaic", "detect", "inject", "photometry", "products", "checks", "stages", "driver")

__all__ = list(MODULES)


def __getattr__(name):
    if name in MODULES:
        return importlib.import_module(__name__ + "." + name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
