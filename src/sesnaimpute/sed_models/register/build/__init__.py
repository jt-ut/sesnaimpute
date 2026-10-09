"""
build/__init__.py
====================================================================
The scientific core of register assembly: read a class's member
libraries, derive their per-library terms, build the shared quotient-
space projector, and (for a pooled class) assemble the joint register
directly.

Trimmed, 2026-10-08 curation consolidation: the production-receipt
layer this module used to serve (`LibrarySpec` plumbing, per-key
output-path staging, dry-run, `build_pooled`'s read-back-from-already-
written-members path, `show`) is gone with the driver it served
(`sed_models_register/build/registers.py`, deleted whole). What
remains is exactly the numerics `sesnaimpute.sed_models.library.
write_register` calls: `_derive_materials` (read every member, derive
per-library terms, build the shared projector) and `build_pooled_direct`
(assemble one pooled register from a class's members without ever
writing or reading back an individual member container).

[Curation density deletion, this branch] `density.derive` -- the
per-CLASS grid-density quadrature estimator that used to run here,
feeding RHO_K/RHO_2K/RHO_KDE1/RHO_KDE2/R_1/BOUNDARY -- is GONE: the old
design's library-density weighting, forbidden by the current spec,
read by nothing outside `studies/` and the retired `sed_fit`, and 23 of
the YSO register line's 42 minutes / most of its 20 GB peak. `density.
build_quotient_space()` SURVIVES (it is not what was expensive -- one
SVD of an 8x3 matrix): it still defines QDIM/SIGEFF/SIGPLAC/SIGLGSRC
and the `/bands` table's AV_LAW_* columns, and nothing else in this
module depends on a per-class pooling pass any more, so classes no
longer need to be read together before any of them can be written.

WHERE EACH KIND OF FACT LIVES:

  canon     every CLASS and SUBCLASS NAME -- constants.CLASSMAP. Never
            redeclared locally; a library shipping a name outside it is
            a curation bug to fix at the source, not to alias around.
  curator   what each model IS -- SUBCLASS, read verbatim from the
            library's own classmap.fits.
  caller    which libraries belong to one class, and where they are --
            a `(key, cls, path)` triple per library, in whatever
            sequence type the caller likes (duck-typed, see
            `_derive_materials`).
  here      how to turn those into containers.

THE SAMPLING SCALE IS NOT A PARAMETER HERE ANY MORE. `density.
build_quotient_space()` (no `sigma_log` argument) now DEFAULTS to
`constants.LIBRARY_SAMPLING_SIGMA_LOG_VECTOR` -- the owner's ruling
scale, a literature constant with its derivation in that constant's own
docstring, not a quantity measured from the catalogue. There is
therefore no placeholder mode, no `sigma_log_path` to thread through,
and no `sed_models_register.noise` product to read (that catalogue-
measured artefact is retired; its own source files are gone).
====================================================================
"""

import os
import time

from sesnaimpute.sed_models.constants import SUBCLASSES_OF

from ..classes import Registry
from ..library import ModelLibrary
from .. import density, io, reference, synth

__all__ = ["VOCAB", "_derive_materials", "build_pooled_direct"]

#: Read from the canon, never declared here -- a local copy could only
#: drift from constants.CLASSMAP, and an alias table reconciling the two
#: is exactly what the canon exists to prevent.
VOCAB = {c: set(subs) for c, subs in SUBCLASSES_OF.items()}


def _noop(_msg):
    pass


def _derive_materials(specs, *, log=print):
    """Stages 1-2, shared by every build entry point: read each library
    and derive its per-library terms.

    `specs` is any sequence of objects with `.key`, `.cls`, `.path`
    attributes (duck-typed; the caller decides the concrete type).

    Extracted so `build_pooled_direct` can run the exact same derivation
    for a class's members WITHOUT the write stage -- letting a pooled
    root register be assembled from the per-library results without
    ever writing (or reading back) an individual member container.

    Returns
    -------
    (space, keys, libs, cmaps, refs, synths, rows, t0)
        `keys` is `specs`' order; the rest are dicts keyed by library
        key. `t0` is a `time.perf_counter()` start, for a caller's own
        "done in Ns" line.
    """
    log = log or _noop
    specs = list(specs)
    keys = [s.key for s in specs]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ValueError(f"duplicate library keys: {dupes}")
    for s in specs:
        if not os.path.isdir(s.path):
            raise ValueError(f"{s.key}: no such directory {s.path!r}")

    reg = Registry({s.key: s.cls for s in specs},
                   {s.key: s.path for s in specs}, VOCAB)
    paths = {s.key: s.path for s in specs}
    LIBRARIES = {s.key: s.cls for s in specs}

    t0 = time.perf_counter()
    libs, refs, synths, cmaps = {}, {}, {}, {}
    for k in keys:
        t = time.perf_counter()
        lib = ModelLibrary(paths[k])
        lib.check_alignment()
        cm = lib.classmap
        declared = set(cm["class_declared"].tolist())
        if declared != {LIBRARIES[k]}:
            raise ValueError(
                f"{k}: classmap.fits declares CLASS={sorted(declared)}, the "
                f"caller's spec assigns {LIBRARIES[k]!r}. These must agree. "
                f"Fix the library's classmap or the caller -- do NOT add an "
                f"alias; one canonical vocabulary is the point."
            )
        bad = sorted(set(cm["subclass"].tolist()) - VOCAB[LIBRARIES[k]])
        if bad:
            raise ValueError(
                f"{k}: SUBCLASS values {bad} are outside constants.CLASSMAP "
                f"for class {LIBRARIES[k]}. Either the library added a "
                f"category or the canon is stale -- resolve at the source."
            )
        libs[k], cmaps[k] = lib, cm
        refs[k] = reference.derive(lib)
        synths[k] = synth.derive(lib)
        log(f"  {k:<14s} n={lib.meta.n_models:>6d}  "
            f"[{time.perf_counter() - t:5.1f}s]")

    log("")
    space = density.build_quotient_space()
    log(f"quotient space: d={space.d}  sigma_eff={space.sigma_eff:.9f}")

    rows = {r["LIBRARY"]: r for r in reg.provenance_records()}
    return space, keys, libs, cmaps, refs, synths, rows, t0


def build_pooled_direct(specs, *, key, member_keys, output_path, note=None, log=print):
    """Assemble one pooled root register DIRECTLY from `specs` -- the
    joint per-class derivation -- without ever writing or reading back
    an individual member container.

    Parameters
    ----------
    specs : sequence of objects with `.key`, `.cls`, `.path`
        Every member of ONE pooled class (e.g. YSO's five strata).
        `set(member_keys)` must equal `{s.key for s in specs}`.
    key : str
        The pooled register's own LIBRARY/output-naming key (e.g.
        `"yso"`).
    member_keys : tuple
        Declared member order -- the pooled `/models` table's row order.
    output_path : str
        Where to write the pooled container.
    note : str, optional
        Free-text provenance, carried into the pooled COMMENT block
        (`io.write_pooled`'s `note`).
    log : callable
        Progress sink; pass None to silence.

    Returns
    -------
    The output path.
    """
    log = log or _noop
    space, keys, libs, cmaps, refs, synths, rows, t0 = (
        _derive_materials(specs, log=log))
    if set(member_keys) != set(keys):
        raise ValueError(
            f"build_pooled_direct({key!r}): member_keys {sorted(member_keys)} "
            f"does not match the specs given {sorted(keys)}"
        )
    member_materials = [
        (mk, dict(lib=libs[mk], classmap=cmaps[mk], reference_state=refs[mk],
                  synth_tables=synths[mk],
                  quotient_space=space, registry_row=rows[mk]))
        for mk in member_keys
    ]
    out = os.fspath(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    io.write_pooled_direct(out, key=key, member_materials=member_materials,
                           note=note)
    chk = io.read(out)
    log(f"  wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB, "
        f"{len(chk['models'])} rows, read-back OK)"
        f"\ndone in {time.perf_counter() - t0:.1f}s")
    return out
