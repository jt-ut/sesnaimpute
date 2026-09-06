"""The fit driver: batches a region's sources through the class posterior's
evidence sweep and joins the batches into one evidence file per region and
class (`10_POSTERIOR.md` section 1's `L_hat`/`Gamma`/`Psi` sweep;
`reading/06_fitter_and_impute.md` section C -- the old `sed_fit.batch`
batch loop, lifted without its provenance, fingerprint or schema-validator
machinery).

For one region and one class, the sources are read in batches of
`BATCH_SIZE` (the SED fitter's own long-standing default,
`sesnacomplete.sed_fit.batch.generate_batch_ranges`'s `batch_size=10000`,
the unit `CODING_RULES.md` 10b and `prior.callable.SourcePrior.prepare`'s
own per-batch tabulation are both sized to). The batches of one {region,
class} are dispatched to `config.n_jobs` `loky` worker processes
(`joblib.Parallel`, module docstring's "PARALLEL DISPATCH" section below)
-- separate processes, not threads, so each worker's own numpy calls stay
single-threaded and do not oversubscribe the machine's cores. Each worker
builds its own `SourcePrior`/`GaiaTerm`/`PsiTerm` (they hold open HDF5
handles and per-region tabulations that cannot cross a process boundary),
calls `prior.prepare(rows)` for its batch, sweeps it with `fit.sweep.
fit_batch`, and writes its own batch file. Once every batch of a {region,
class} has landed, they are joined into one evidence file in catalogue
row order -- the same read-all-write-once join `sed_fit.io.
assemble_fitres` performs -- and the batch files are removed.

PARALLEL DISPATCH AND ITS MEMORY BUDGET (item 1). Two nested budgets
apply, both already governed by existing sizing rules, multiplied by
`n_jobs`:

- `fit.sweep.fit_batch`'s own inner block (its module docstring): a
  `(n_block, n_model, n_band)` float64 working set held under 512 MB via
  `sesnaimpute.batches.batches(n_source, row_bytes)`, `row_bytes = n_model
  * n_band * 8 bytes/float64 * 8` (about eight such arrays alive at once
  -- residual, model, chi2 and their block-sized cousins). For the
  largest register, YSO (`n_model = 200,000`, `n_band = 8`): `row_bytes =
  200,000 * 8 * 8 * 8 = 102,400,000` bytes (~97.8 MB/source), so `n_block
  = 512 MiB // 97.8 MiB = 5` sources per inner block -- YSO's outer
  10,000-source batch is swept 5 sources at a time internally, capped at
  512 MB regardless of `n_jobs`.
- `SourcePrior.prepare(rows)`'s per-batch tabulation (its own docstring):
  three float32 `EPS` arrays gathered for `BATCH_SIZE = 10,000` rows,
  bytes-per-source bounded by the selection grids' own small axes (`X_
  LADDER` has 8 points; the class/band grids are tens of points), so this
  is tens of MB per batch, independent of the class's register size.

  So one worker's transient peak is dominated by the 512 MB sweep block
  (worst case, any class) plus a few tens of MB of prior tabulation and
  register arrays; `n_jobs` such workers running at once (the parallel
  dispatch this file now does) sit near `n_jobs * (512 MB + tens of MB)`
  -- at `n_jobs = 4` (`root.cfg`'s `[run] n_jobs`), about 2.1-2.5 GB,
  comfortably under the 8 GB ceiling (`CODING_RULES.md` 10a) with room
  for the `SourcePrior`/register/evidence-array overhead measured in the
  timing run below.
"""

import os
import time

import h5py
import numpy as np
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import regions as regions_module
from sesnaimpute.fit.psi import PsiTerm
from sesnaimpute.fit.sweep import fit_batch
from sesnaimpute.fit.terms import GaiaTerm
from sesnaimpute.granules import access
from sesnaimpute.prior.callable import SourcePrior

#: Sources per batch (module docstring): the SED fitter's own
#: long-standing default.
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


#: One process's own `(SourcePrior, GaiaTerm, PsiTerm, native_flux)` for
#: the {region, class} it is currently sweeping (module docstring's
#: "PARALLEL DISPATCH"): built on first use inside that `loky` worker,
#: then reused for every further batch `joblib` hands that same worker
#: within one `_fit_one` call -- each of these objects holds open HDF5
#: reads and a region-wide tabulation that cannot be pickled across a
#: process boundary, so every worker must build its own, but only once.
_WORKER_STATE = {}


def _worker_state(config, region, cls):
    """This worker process's own `(prior, gamma, psi, gaia_cls)` for
    `{region, cls}`, building it on first use and caching it under
    `_WORKER_STATE` (this function's own docstring above) for later
    batches this same worker draws in the same sweep."""
    key = (config.data_root, region, cls)
    state = _WORKER_STATE.get(key)
    if state is None:
        prior = SourcePrior(config, region)
        gaia = GaiaTerm(config, region)
        psi_term = PsiTerm(config, cls)
        native_flux = _library_native_flux(config, cls)
        gaia_cls = cls.lower()

        def gamma(row, model_index, a, log10_b):
            return gaia.ln_gamma(row, model_index, a, log10_b, gaia_cls)

        def psi(row, model_index, a, log10_b):
            return psi_term.ln_psi(native_flux, model_index, a, log10_b)

        state = (prior, gamma, psi)
        _WORKER_STATE.clear()  # one {region, class} live per worker at a time
        _WORKER_STATE[key] = state
    return state


def _run_one_batch(config, region, cls, i, start, stop):
    """One batch, run inside a `loky` worker process (module docstring):
    this worker's own `SourcePrior` is prepared for `[start, stop)`'s
    rows, the class's library is swept against them
    (`fit.sweep.fit_batch`), and the result is written to that batch's
    own file (`_batch_path`), matching the serial path bit for bit --
    `PsiTerm`'s default `beta=0` (its own module docstring) makes `psi`
    the identity, matching `impute.posterior`'s own `BETA = 0`.
    """
    prior, gamma, psi = _worker_state(config, region, cls)
    rows = np.arange(start, stop)
    prior.prepare(rows)
    arrays = fit_batch(config, region, cls, rows, prior, gamma=gamma, psi=psi)
    path = _batch_path(config, region, cls, i)
    _write_batch(path, arrays)
    return path


def _fit_one(config, region, cls):
    """Sweeps one {region, class} in batches of `BATCH_SIZE`, dispatched
    across `config.n_jobs` `loky` worker processes (module docstring's
    "PARALLEL DISPATCH"), then joins the batch files into one evidence
    file in catalogue row order."""
    n_sources = access.region_slice(config, region)["n_sources"]
    bounds = [(i, start, min(start + BATCH_SIZE, n_sources))
              for i, start in enumerate(range(0, n_sources, BATCH_SIZE))]

    batch_paths = Parallel(n_jobs=config.n_jobs, backend="loky")(
        delayed(_run_one_batch)(config, region, cls, i, start, stop)
        for i, start, stop in bounds)

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
