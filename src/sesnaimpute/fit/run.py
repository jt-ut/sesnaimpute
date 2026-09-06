"""The fit driver: batches a region's sources through the class posterior's
evidence sweep and joins the batches into one evidence file per region and
class (`10_POSTERIOR.md` section 1's `L_hat`/`Gamma`/`Psi` sweep;
`reading/06_fitter_and_impute.md` section C -- the old `sed_fit.batch`
batch loop, lifted without its provenance, fingerprint or schema-validator
machinery).

One job is one {region, class}: the census prior for that one class
(`prior.callable.SourcePrior(config, region, cls)`, class-scoped so a
job never tabulates the other five classes' material), the Gaia
congruence term and the colour-cascade term are each built exactly once,
before any source is swept, and held for the whole job -- never rebuilt
per batch. The region's sources are then read in batches of `BATCH_SIZE`
(the SED fitter's own long-standing default,
`sesnacomplete.sed_fit.batch.generate_batch_ranges`'s `batch_size=10000`,
the unit `CODING_RULES.md` 10b and `prior.callable.SourcePrior.prepare`'s
own per-batch tabulation are both sized to): each batch calls `prior.
prepare(rows)` (sequentially -- it mutates the one shared prior's batch
tabulation, so two batches must never prepare it at once), then `fit.
sweep.fit_batch` sweeps the class's library against the batch. The
parallelism (`CODING_RULES.md` 10a's `n_jobs`) lives inside `fit_batch`
itself: once a batch is prepared, every block of that batch's sources is
an independent read against the one shared, already-prepared prior, so
`fit_batch` dispatches its blocks across `config.n_jobs` THREADS (`fit.
sweep`'s own module docstring) rather than processes -- the prior, the
Gaia/Psi terms and the register arrays are read, not copied, by every
thread. Each batch's arrays are written to their own file; once every
batch of a job has landed, they are joined into one evidence file in
catalogue row order -- the same read-all-write-once join `sed_fit.io.
assemble_fitres` performs -- and the batch files are removed.
"""

import os
import time

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import regions as regions_module
from sesnaimpute.fit.psi import PsiTerm
from sesnaimpute.fit.sweep import fit_batch
from sesnaimpute.fit.terms import GaiaTerm
from sesnaimpute.granules import access
from sesnaimpute.prior.callable import SourcePrior

#: Sources per batch (module docstring): the SED fitter's own
#: long-standing default -- a fallback only; `_fit_one` reads the live
#: value off `config.fit_batch_size` (owner ruling 2026-09-06,
#: `config.py`'s `[fit] batch_size`).
BATCH_SIZE = 10_000

CLASSES = tuple(c.code for c in definitions.CLASSES)

#: The one installation config path (`RUNBOOK.sh`'s own `CONFIG=`), used
#: only to fill in `jobs()`'s generated command lines.
CONFIG_PATH = "/Users/jtaylor/Dropbox/Research/SESNA_Complete/config/root.cfg"


def _batch_path(config, region, cls, i):
    """One batch's own file: `evidence-batch_<cls>_<i>` as the product's
    quantity, so the class and batch index sit in the file name
    (module docstring)."""
    quantity = "evidence-batch_%s_%d" % (cls, i)
    return config_module.product_path(config, "bms", "fit", quantity, "source", region=region)


def evidence_path(config, region, cls):
    """The region's one joined evidence file for `cls`, `evidence_<cls>`,
    so the six classes' joined products sit side by side without
    collision -- the one path `impute.posterior` also reads."""
    quantity = "evidence_%s" % cls
    return config_module.product_path(config, "bms", "fit", quantity, "source", region=region)


def _write_batch(path, arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, arr in arrays.items():
            f.create_dataset(key, data=np.asarray(arr))


def _library_native_flux(config, cls):
    """`(n_model, 8)`: this class's library models' own native flux (mJy,
    `definitions.BANDS` order, zero extinction, the library's own
    brightness reference) -- the `rows` argument `fit.psi.PsiTerm.ln_psi`
    reads its model colours off, read once per class straight from its
    register's `F_REF_<band>` columns (`definitions.CLASS_REGISTER`, the
    same key `fit.sweep` reads its own template flux from)."""
    key = definitions.CLASS_REGISTER[cls]
    path = os.path.join(config.inputs["sed_models"], "registers", "%s_register.hdf5" % key)
    with h5py.File(path, "r") as f:
        n_model = f["models"]["MODEL_NAME"].shape[0]
        flux = np.empty((n_model, len(definitions.BANDS)), dtype=np.float64)
        for j, band in enumerate(definitions.BANDS):
            flux[:, j] = np.asarray(f["models"]["F_REF_%s" % band.key][:], dtype=np.float64)
    return flux


def _fit_one(config, region, cls):
    """Sweeps one {region, class} job (module docstring): the class's
    `SourcePrior`, `GaiaTerm` and `PsiTerm` are built once, here, before
    the batch loop, and held for every batch of this job; each batch of
    `BATCH_SIZE` sources is prepared sequentially and swept by `fit.
    sweep.fit_batch` (which does its own thread-parallel work inside);
    the batch files are then joined into one evidence file in catalogue
    row order.
    """
    n_sources = access.region_slice(config, region)["n_sources"]
    # SourcePrior's own class vocabulary is lower-case (`prior.callable.
    # CLASSES`); `definitions.CLASSES` codes (this module's own `cls`,
    # `CLASSES` above) are upper-case -- lower() at the one boundary
    # where a fitter job hands its class to the prior.
    prior = SourcePrior(config, region, cls.lower())
    gaia = GaiaTerm(config, region)
    psi_term = PsiTerm(config, cls, beta=config.fit_beta)
    native_flux = _library_native_flux(config, cls)
    gaia_cls = cls.lower()
    batch_size = config.fit_batch_size

    def gamma(row, model_index, a, log10_b):
        return gaia.ln_gamma(row, model_index, a, log10_b, gaia_cls)

    def psi(row, model_index, a, log10_b):
        return psi_term.ln_psi(native_flux, model_index, a, log10_b)

    batch_paths = []
    for i, start in enumerate(range(0, n_sources, batch_size)):
        rows = np.arange(start, min(start + batch_size, n_sources))
        prior.prepare(rows)
        arrays = fit_batch(config, region, cls, rows, prior, gamma=gamma, psi=psi)
        path = _batch_path(config, region, cls, i)
        _write_batch(path, arrays)
        batch_paths.append(path)

    joined = _join(config, region, cls, batch_paths, n_sources)
    for path in batch_paths:
        os.remove(path)
    return joined


def _join(config, region, cls, batch_paths, n_sources):
    """Concatenates every batch file's own datasets, in the batches'
    catalogue-row order (`_fit_one`'s own `range(0, n_sources,
    BATCH_SIZE)`), verifies the joined row count against the region's
    own catalogued source count, and writes the one per-class evidence
    file with `GRANULE`/`CLASS` root attributes (module docstring)."""
    per_batch = []
    for path in batch_paths:
        with h5py.File(path, "r") as f:
            per_batch.append({key: f[key][:] for key in f.keys()})
    joined = {key: np.concatenate([b[key] for b in per_batch], axis=0)
              for key in per_batch[0].keys()}

    n_written = next(iter(joined.values())).shape[0]
    if n_written != n_sources:
        raise ValueError(
            "fit.run: %r/%r joined %d rows, region has %d catalogued sources"
            % (region, cls, n_written, n_sources))

    out_path = evidence_path(config, region, cls)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["GRANULE"] = "source"
        f.attrs["CLASS"] = cls
        for key, arr in joined.items():
            f.create_dataset(key, data=arr)
    return out_path


def build(config, regions=None, classes=None):
    """Sweeps every (region, class) pair, batched (module docstring):
    `regions` defaults to `constants.REGIONS` (all thirty, `CODING_RULES.md`
    5c), `classes` to the six `definitions.CLASSES` codes."""
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    class_codes = classes if classes is not None else list(CLASSES)
    for region in region_names:
        for cls in class_codes:
            t0 = time.time()
            out_path = _fit_one(config, region, cls)
            print("fit.run: %s/%s -> %s (%.1fs)" % (region, cls, out_path, time.time() - t0),
                  flush=True)


def jobs(config):
    """Writes `bms/fit/jobs.sh`, one line per (region, class) job so a
    cluster runs every job in parallel (module docstring): `./capped.sh
    python -m sesnaimpute.fit.run <config> --regions R --classes C`."""
    region_names = [r.name for r in regions_module.REGIONS]
    out_path = os.path.join(config.data_root, "bms", "fit", "jobs.sh")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        for region in region_names:
            for cls in CLASSES:
                f.write('./capped.sh python -m sesnaimpute.fit.run %s --regions "%s" --classes %s\n'
                        % (CONFIG_PATH, region, cls))
    os.chmod(out_path, 0o755)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--classes", nargs="+", default=None)
    args = parser.parse_args()
    cfg = config_module.load(args.config)
    build(cfg, regions=args.regions, classes=args.classes)
