"""
classes.py
====================================================================
The registry: which libraries make up which class, and what their
subclass labels are.

`library.py` handles ONE library -- its files, fluxes, cube, and labels.
This module handles the COLLECTION: it validates the labels against a
declared vocabulary, groups libraries into classes, and produces the
provenance record that travels into every posterior.fits.

DATA LIVES IN THE DRIVER, NOT HERE. This module contains no library
names, no class names, no paths and no vocabulary -- the driver supplies
them, so the whole run configuration stays readable in one file rather
than buried in package code. What lives here is the machinery that
consumes it. The subclass VOCABULARY itself is `constants.CLASSMAP`,
the project's one canonical record of every class and subclass name.

Three deliberate properties:

1. VALIDATION IS FAIL-LOUD AND EXHAUSTIVE. A subclass value outside the
   declared vocabulary is a build failure, not a passthrough -- that is
   the check that catches a curator adding a category without telling
   anyone, which has already happened twice in this project. And all
   violations are reported together rather than one per run, because a
   build sweeps nine libraries and finding them one at a time is nine
   round trips.

2. THE CLASS IS THE POOLING UNIT. `class_groups()` is what density.py
   consumes to know that YSO spans five directories and must be measured
   as one pooled set (N=199,999, k=227) rather than five separate ones,
   whose k values would differ by 3x and whose densities would not be
   comparable. It is also why reassigning a library to another class is a
   one-line edit in the driver: the repooling follows automatically.

3. THE PROVENANCE RECORD travels into every container, so a census
   product is self-describing without the libraries present. It is
   written by io.py rather than kept in a side-car file that could fall
   out of sync with what was actually built.
====================================================================
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .library import ModelLibrary

__all__ = ["Registry", "ResolvedLibrary", "ClassGroup", "ValidationError"]


class ValidationError(RuntimeError):
    """Raised when the registry does not match what is on disk.

    Carries every violation found, not just the first.
    """

    def __init__(self, problems):
        self.problems = list(problems)
        body = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(
            f"{len(self.problems)} registry problem(s):\n{body}"
        )


@dataclass(frozen=True)
class ResolvedLibrary:
    """One library after its labels have been read and checked."""

    key: str
    path: str
    cls: str
    n_models: int
    aperture_dependent: bool
    subclass: np.ndarray          # (n_models,) stripped strings
    source: str                   # where the labels came from
    counts: Mapping[str, int]     # subclass token -> count, sorted

    @property
    def summary(self):
        return "; ".join(f"{k}:{v}" for k, v in self.counts.items())


@dataclass(frozen=True)
class ClassGroup:
    """One class, pooled across however many libraries compose it.

    `n_models` is the POOLED count -- the N that sets k(N) for the kNN
    density estimate, and the population the quadrature weight is
    normalized over.
    """

    cls: str
    library_keys: tuple
    n_models: int
    vocab: frozenset

    @property
    def is_pooled(self):
        return len(self.library_keys) > 1


class Registry:
    """The driver's mapping, resolved and validated against disk.

    Parameters
    ----------
    libraries : mapping of library key -> census class name
    paths : mapping of library key -> filesystem path
    vocab : mapping of class name -> allowed subclass tokens
    """

    def __init__(self, libraries, paths, vocab):
        self.libraries = dict(libraries)
        self.paths = dict(paths)
        self.vocab = {k: frozenset(v) for k, v in vocab.items()}
        self._resolved = None

    # ------------------------------------------------------------ resolve

    def resolve(self, force=False):
        """Read and validate every library's labels.

        Raises ValidationError listing ALL problems found. Caches on
        success; pass force=True to re-read.
        """
        if self._resolved is not None and not force:
            return self._resolved

        problems = []
        resolved = []

        missing_path = set(self.libraries) - set(self.paths)
        for key in sorted(missing_path):
            problems.append(f"{key}: has a class but no path")

        orphan_path = set(self.paths) - set(self.libraries)
        for key in sorted(orphan_path):
            problems.append(
                f"{key}: has a path but no class -- a library the registry "
                f"does not classify would be silently skipped"
            )

        for key in sorted(set(self.libraries) & set(self.paths)):
            cls = self.libraries[key]
            if cls not in self.vocab:
                problems.append(f"{key}: class {cls!r} has no VOCAB entry")
                continue
            try:
                lib = ModelLibrary(self.paths[key])
            except Exception as exc:
                problems.append(f"{key}: cannot open {self.paths[key]!r} -- {exc}")
                continue

            # SUBCLASS comes from the library's own classmap.fits and is
            # read verbatim -- it says what the models ARE, which is the
            # curator's call. Only the CENSUS class is ours.
            try:
                cm = lib.classmap
                labels = cm["subclass"]
                source = "classmap.fits SUBCLASS"
            except Exception as exc:
                problems.append(f"{key}: classmap.fits unreadable -- {exc}")
                continue

            n = lib.meta.n_models
            labels = np.asarray(labels).astype(str)
            labels = np.char.strip(labels)

            if len(labels) != n:
                problems.append(
                    f"{key}: classmap.fits has {len(labels)} rows for "
                    f"{n} models"
                )
                continue

            blank = int(np.count_nonzero(labels == ""))
            if blank:
                problems.append(
                    f"{key}: {blank} of {n} models have a blank subclass"
                )

            found = set(np.unique(labels).tolist())
            allowed = self.vocab[cls]
            undeclared = sorted(found - allowed - {""})
            if undeclared:
                problems.append(
                    f"{key}: subclass value(s) {undeclared} not in the "
                    f"declared vocabulary for class {cls!r} "
                    f"({sorted(allowed)})"
                )

            values, counts = np.unique(labels, return_counts=True)
            resolved.append(ResolvedLibrary(
                key=key,
                path=self.paths[key],
                cls=cls,
                n_models=n,
                aperture_dependent=lib.meta.aperture_dependent,
                subclass=labels,
                source=source,
                counts={str(v): int(c) for v, c in zip(values, counts)},
            ))

        if problems:
            raise ValidationError(problems)

        # Declared-but-unseen vocabulary is a warning-shaped fact, not a
        # failure: a class may legitimately have a token no model uses yet.
        # It is recorded rather than raised, and surfaces in show().
        self._resolved = tuple(resolved)
        return self._resolved

    # ------------------------------------------------------------ grouping

    def class_groups(self):
        """Class name -> ClassGroup, with pooled model counts.

        This is the density-pooling unit. A class spanning several
        libraries is measured as one population.
        """
        out = {}
        for r in self.resolve():
            keys, n = out.get(r.cls, ((), 0))
            out[r.cls] = (keys + (r.key,), n + r.n_models)
        return {
            cls: ClassGroup(cls=cls, library_keys=keys, n_models=n,
                            vocab=self.vocab[cls])
            for cls, (keys, n) in sorted(out.items())
        }

    def unused_vocabulary(self):
        """Declared tokens that no model actually carries, per class."""
        seen = {}
        for r in self.resolve():
            seen.setdefault(r.cls, set()).update(r.counts)
        return {
            cls: sorted(self.vocab[cls] - seen.get(cls, set()))
            for cls in sorted(self.vocab)
            if self.vocab[cls] - seen.get(cls, set())
        }

    # ------------------------------------------------------------ records

    def provenance_records(self):
        """Flat, serializable rows for posterior.fits's LIBRARY HDU.

        Returned as plain dicts; the build core owns the container encoding, so this
        module stays independent of the storage format.
        """
        return [
            {
                "LIBRARY": r.key,
                "CLASS": r.cls,
                "N_MODELS": r.n_models,
                "APERTURE_DEPENDENT": bool(r.aperture_dependent),
                "SUBCLASS_SOURCE": r.source,
                "SUBCLASS_VALUES": r.summary,
                "PATH": r.path,
            }
            for r in self.resolve()
        ]

    # ------------------------------------------------------------ display

    def show(self):
        """Human-readable rendering of the resolved mapping.

        The driver exposes this behind --show so the mapping can be
        eyeballed before a build, without opening any FITS file.
        """
        rows = self.resolve()
        groups = self.class_groups()
        w_key = max((len(r.key) for r in rows), default=7)
        w_src = max((len(r.source) for r in rows), default=6)

        lines = ["Resolved class / subclass registry", ""]
        lines.append(f"  {'library':<{w_key}}  {'class':<5}  {'n':>7}  "
                     f"{'subclass source':<{w_src}}  values")
        lines.append("  " + "-" * (w_key + w_src + 34))
        for r in sorted(rows, key=lambda r: (r.cls, r.key)):
            lines.append(
                f"  {r.key:<{w_key}}  {r.cls:<5}  {r.n_models:>7d}  "
                f"{r.source:<{w_src}}  {r.summary}"
            )

        lines += ["", "Class groups (the density-pooling unit):"]
        for cls, g in groups.items():
            note = (f"pooled over {len(g.library_keys)} libraries"
                    if g.is_pooled else "single library")
            k = max(10, round(g.n_models ** (4 / 9)))
            lines.append(
                f"  {cls:<5}  N={g.n_models:>7d}  k(N)={k:<4d}  {note}"
            )

        unused = self.unused_vocabulary()
        if unused:
            lines += ["", "Declared but unused vocabulary (not an error):"]
            for cls, toks in unused.items():
                lines.append(f"  {cls:<5}  {toks}")
        return "\n".join(lines)
