"""The PACS completeness of a region in stages, each saved when it is done, so that a killed job loses only the file
it was writing and the work of one region can be spread over many jobs (`pacs_completeness` is the measurement, this
module the order and the files).

The stage store, `stage_dir(config, region, box)` = `<data_root>/sky/derived/pacs/stages/<region>[__slice_<box>]/`, one
subdirectory per band holding (pickle files, each written under a temporary name and renamed into place: a stage is
done when its file exists):

    detect.pkl                       what the band holds before its artificial sources: cells, noise, the PACS sources,
                                     the matching radius, the first round's size per map
    inject_r<r>_s<k>of<K>.pkl        the outputs of shard k (of K) of the pieces of injection round r
    inject_r<r>_s<k>of<K>.counts.pkl the injections that shard added to each cell's count (what round r + 1 is chosen from)
    pool_s<k>of<K>.pkl               the outputs of shard k (of K) of the pooling blocks
    band.pkl                         the band's datasets (`pacs_completeness._band_dict`), the held-out half included

and `held_<band>.pkl` beside the band directories, the band's held-out half (what `acceptance` reads). The pieces of
a round, and of the pooling, are a deterministic list (`round_tasks`, `pool_task_list`): shard k takes `[k::K]` of it,
every piece seeded by its own (block, S/N plane, half, batch), so the product does not depend on K or on which job did
which shard. Round r + 1 is planned from the counts of every earlier round (a barrier: all shards of round r first).
Run: `run_region` for all of it in one process, or one stage at a time.
"""

import functools
import os
import pickle
import re
import shutil


from sesnaimpute import pieces as pieces_module
from sesnaimpute.config import product_path
from sesnaimpute.sky.derived import pacs_completeness as pc
from sesnaimpute.sky.derived import pacs_fit


# ----------------------------------------------------------------------------
# the store

def stage_dir(config, region, box=None):
    """The region's stage store (named as `pacs_mosaic.product_dir` names a mosaic)."""
    safe = re.sub(r"[^A-Za-z0-9+-]+", "_", region)
    return os.path.join(config.data_root, "sky", "derived", "pacs", "stages",
                        safe + ("" if box is None else "__slice_%.3f_%.3f_%.2fx%.2f" % tuple(box)))


def _band_dir(config, region, band, box=None):
    d = os.path.join(stage_dir(config, region, box), band)
    real, root = os.path.realpath(d), os.path.realpath(os.path.join(config.data_root, "sky", "derived", "pacs"))
    assert real.startswith(root + os.sep), "%s resolves outside %s" % (d, root)
    os.makedirs(d, exist_ok=True)
    return d


def _atomic_write(path, obj):
    """Pickle `obj` (protocol 5) to `path` through a temporary file in the same directory, on disk before the rename
    (Orion A on Stampede3: a 296 MB shard file renamed into place without it read back 280 MiB long)."""
    tmp = "%s.tmp%d" % (path, os.getpid())
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=5)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _shard_name(stem, shard, n_shards):
    return "%s_s%dof%d.pkl" % (stem, shard, n_shards)


def _shard_files(d, stem):
    """`(K, [paths of shards 0..K-1])` of the files `<stem>_s<k>of<K>.pkl` of a stage, or None when there is none.
    Raises, naming the file, when a shard is missing or two shardings are mixed."""
    found = {}
    for name in os.listdir(d):
        m = re.fullmatch(re.escape(stem) + r"_s(\d+)of(\d+)\.pkl", name)
        if m:
            found.setdefault(int(m.group(2)), set()).add(int(m.group(1)))
    if not found:
        return None
    if len(found) > 1:
        raise ValueError("%s in %s: shardings of %s mixed" % (stem, d, sorted(found)))
    (n_shards, have), = found.items()
    missing = sorted(set(range(n_shards)) - have)
    if missing:
        raise ValueError("%s: shard %d of %d is missing (%s)" % (stem, missing[0], n_shards, os.path.join(d, _shard_name(stem, missing[0], n_shards))))
    return n_shards, [os.path.join(d, _shard_name(stem, k, n_shards)) for k in range(n_shards)]


def _clear_store(config, region, box):
    """The stage store removed, except the held-out halves (`held_<band>.pkl`, which `acceptance` reads)."""
    sd = stage_dir(config, region, box)
    for b in pc.BANDS:
        shutil.rmtree(os.path.join(sd, b), ignore_errors=True)
    if os.path.exists(os.path.join(sd, "finish_submitted")):      # the claim of the finish job (`pacs_driver.claim_finish`)
        os.remove(os.path.join(sd, "finish_submitted"))


def clear_prepared(config, region, box=None):
    """The region's prepared layers removed (`pacs_fit.prepared_dir` of each band's mosaic): two thirds of a region's
    disk (Orion A 17 of 25 GB; all regions together would exceed the /work quota), rebuilt from the mosaic in minutes
    and read by nothing once the photometry is written. The mosaics stay."""
    from sesnaimpute.sky.derived import pacs_mosaic
    for b in pc.BANDS:
        if os.path.exists(os.path.join(pacs_mosaic.product_dir(config, region, b, box), "mosaic.hdf5")):
            d = pacs_fit.prepared_dir(config, pacs_fit.mosaic_record(config, region, b, box))
            real, root = os.path.realpath(d), os.path.realpath(os.path.join(config.data_root, "sky", "derived", "pacs", "prepared"))
            assert real.startswith(root + os.sep), "%s resolves outside %s" % (d, root)
            shutil.rmtree(d, ignore_errors=True)


# ----------------------------------------------------------------------------
# the band's inputs

def _mosaic_record(config, region, band, box, n_jobs):
    """The band's mosaic record, the mosaic built when absent; None where the region has no map of the band."""
    from sesnaimpute.sky.derived import pacs_mosaic
    if not os.path.exists(os.path.join(pacs_mosaic.product_dir(config, region, band, box), "mosaic.hdf5")):
        if not [r for r in pacs_fit.read_index(config, region) if r.band == band]:
            return None
        pacs_mosaic.build_mosaic(config, region, band, box=box, n_jobs=n_jobs)
    return pacs_fit.mosaic_record(config, region, band, box)


def prepare(config, region, box=None, n_jobs=None, stage=None):
    """The mosaics of the region built, their layers and detection kernels prepared, and every band's self-checks, all
    before any band's injection (a P100 or P160 failure is minutes in, not after the P70 hour). Returns
    `{band: record}` of the bands the region has."""
    from sesnaimpute.sky.derived import pacs_checks
    n_jobs = n_jobs or config.n_jobs
    records = {}
    for b in pc.BANDS:
        rec = _mosaic_record(config, region, b, box, n_jobs)
        if rec is not None:
            records[b] = rec
    pc._prepare_stage(config, list(records.values()), n_jobs, stage)
    for rec in records.values():
        pc.self_check(config, rec)
        pacs_checks.normalisation_check(config, rec)
    return records


def _detect_state(config, region, band, box):
    path = os.path.join(_band_dir(config, region, band, box), "detect.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("%s: stage detect of %s %s has not run" % (path, region, band))
    return _read(path)


def _nx(s):
    return {t: s["geometry"][t][2].nx for t in s["inputs"]}


# ----------------------------------------------------------------------------
# the stages

def detect(config, region, band, box=None, n_jobs=None, stage=None):
    """Stage DETECT of one band: the cells and their noise, the PACS sources, the matching radius and the size of each
    map's first injection round, written to `detect.pkl`. A band the region has no map of (or no source in reach) gets
    the marker `dict(empty=True, n_src=...)`."""
    path = os.path.join(_band_dir(config, region, band, box), "detect.pkl")
    if os.path.exists(path):
        return
    n_jobs = n_jobs or config.n_jobs
    mag, flux_grid = pc.canonical_grid()
    ra, dec = pc._region_inputs(config, region)
    record = _mosaic_record(config, region, band, box, n_jobs)
    s = None
    if record is not None:
        pc._prepare_stage(config, [record], n_jobs, stage)
        s = pc._band_detect(config, [record], ra, dec, n_jobs, stage, flux_grid)
    if s is None:
        _atomic_write(path, dict(empty=True, n_src=int(ra.size)))
        return
    s["first"] = pc.map_sizing(config, s["records"], s["inputs"], n_jobs, stage)
    _atomic_write(path, s)


def _round_loader(d, s):
    """`load(q)` of `pc.round_task_list`: the summed counts of round q's shards (the counts files)."""
    def load(q):
        shards = _shard_files(d, "inject_r%d" % q)
        if shards is None:
            raise ValueError("injection round %d of %s has no shard files: it has not run" % (q, d))
        total = {}
        for p in shards[1]:
            for t, m in _read(p[:-len(".pkl")] + ".counts.pkl").items():
                total[t] = total[t] + m if t in total else m
        return total
    return load


def round_tasks(config, region, band, r, box=None):
    """The ordered pieces of injection round `r` of the band ([] = the band's injections are complete). Round r > 0 is
    planned from `detect.pkl` and every inject file of the rounds before it (raises, naming the file, when a shard of
    an earlier round is missing)."""
    s = _detect_state(config, region, band, box)
    if s.get("empty"):
        return []
    d = _band_dir(config, region, band, box)
    return pc.round_task_list(s["records"], s["inputs"], _nx(s), s["first"], r, _round_loader(d, s))


def inject(config, region, band, r, shard, n_shards, box=None, n_jobs=None, stage=None, tasks=None):
    """Stage INJECT: shard `shard` of `n_shards` of round `r`, the pieces `round_tasks(...)[shard::n_shards]`, written
    to `inject_r<r>_s<shard>of<n_shards>.pkl` (nothing is written for a round without pieces). `tasks`: the round's
    `round_tasks`, when the caller has them."""
    d = _band_dir(config, region, band, box)
    path = os.path.join(d, _shard_name("inject_r%d" % r, shard, n_shards))
    if os.path.exists(path):
        return
    tasks = round_tasks(config, region, band, r, box) if tasks is None else tasks
    if not tasks:
        return
    s = _detect_state(config, region, band, box)
    done = pieces_module.run(functools.partial(pc._inject_pairs, config=config, records=s["records"]), tasks[shard::n_shards],
                             n_jobs or config.n_jobs, stage=stage, unit="map pieces (artificial sources, round %d)" % r)
    _atomic_write(path[:-len(".pkl")] + ".counts.pkl", pc.part_counts(s["inputs"], done))
    _atomic_write(path, done)


def pool_task_list(config, region, band, box=None):
    """The pooling blocks of the band, in order (`pc.pool_tasks` of each map)."""
    s = _detect_state(config, region, band, box)
    if s.get("empty"):
        return []
    nx = _nx(s)
    return [task for t in pc.tile_order(s["inputs"])
            for task in pc.pool_tasks(t, pc.pool_dir(config, s["records"][t]), s["inputs"][t]["ids"], nx[t])]


def _injection_parts(d, s):
    """`{tile: [pieces' outputs]}` of every round of the band, read from the inject files; raises when a round that
    had pieces has missing shards or the last round is not the band's last."""
    nx = _nx(s)
    parts, r = {t: [] for t in s["inputs"]}, 0
    while _shard_files(d, "inject_r%d" % r) is not None:
        for p in _shard_files(d, "inject_r%d" % r)[1]:
            for t, part in _read(p):
                parts[t].append(part)
        r += 1
    if pc.round_task_list(s["records"], s["inputs"], nx, s["first"], r, _round_loader(d, s)):
        raise ValueError("%s: injection round %d has pieces but no files: the injections are not complete" % (d, r))
    return parts


def _pool_tables(config, d, s):
    """The memory-mapped injection table of every map of the band (`pc._write_pool_table`, with the held-out half and
    the statistics in `meta.pkl`), each written to a temporary directory and renamed into place by whichever pooling
    shard finds it absent."""
    todo = [t for t in pc.tile_order(s["inputs"]) if not os.path.exists(pc.pool_dir(config, s["records"][t]))]
    if not todo:
        return
    parts = _injection_parts(d, s)
    for t in todo:
        final = pc.pool_dir(config, s["records"][t])
        tmp = "%s.tmp%d" % (final, os.getpid())
        shutil.rmtree(tmp, ignore_errors=True)
        _, held, stats = pc._write_pool_table(t, s["inputs"][t], parts[t], s["records"], config, d=tmp)
        _atomic_write(os.path.join(tmp, "meta.pkl"), (held, stats))
        try:
            os.rename(tmp, final)
        except OSError:
            if not os.path.exists(final):
                raise
            shutil.rmtree(tmp)


def pool_tables(config, region, band, box=None):
    """The band's injection tables written once, before its pooling shards start (the submit chain's own job: K
    shards started together would each hold every injection of the band to write the same tables)."""
    d = _band_dir(config, region, band, box)
    s = _detect_state(config, region, band, box)
    if not s.get("empty"):
        _pool_tables(config, d, s)


def pool(config, region, band, shard, n_shards, box=None, n_jobs=None, stage=None):
    """Stage POOL: shard `shard` of `n_shards` of the pooling blocks (`pool_task_list`), written to
    `pool_s<shard>of<n_shards>.pkl`. The injection tables of the band's maps are written first, by the shard that finds
    them absent (a pool shard waits for nothing: run it after the band's last round)."""
    d = _band_dir(config, region, band, box)
    path = os.path.join(d, _shard_name("pool", shard, n_shards))
    s = _detect_state(config, region, band, box)
    if os.path.exists(path) or s.get("empty"):
        return
    _, flux_grid = pc.canonical_grid()
    tasks = pool_task_list(config, region, band, box)[shard::n_shards]
    if tasks:
        _pool_tables(config, d, s)
    done = pieces_module.run(functools.partial(pc._pool_block, records=s["records"], flux_grid=flux_grid), tasks,
                             n_jobs or config.n_jobs, stage=stage, unit="cell blocks (pooling)")
    _atomic_write(path, done)


def assemble(config, region, band, box=None):
    """Stage ASSEMBLE: the band's datasets from `detect.pkl`, the pooling shards and the injection tables, written to
    `band.pkl` and the held-out half to `held_<band>.pkl`; the injection tables removed after."""
    sd = stage_dir(config, region, box)
    d = _band_dir(config, region, band, box)
    path = os.path.join(d, "band.pkl")
    s = _detect_state(config, region, band, box)
    mag, flux_grid = pc.canonical_grid()
    if not os.path.exists(path):
        if s.get("empty"):
            band_dict = pc._empty_band(s["n_src"], mag.size)
        else:
            shards = _shard_files(d, "pool")
            if shards is None:
                raise ValueError("%s: stage pool has not run" % d)
            blocks_of = {t: [] for t in s["inputs"]}
            n_blocks = 0
            for p in shards[1]:
                for t, o in _read(p):
                    blocks_of[t].append(o)
                    n_blocks += 1
            if n_blocks != len(pool_task_list(config, region, band, box)):
                raise ValueError("%s: the pooling shards hold %d blocks, the band has %d" % (d, n_blocks, len(pool_task_list(config, region, band, box))))
            injected = []
            for t in pc.tile_order(s["inputs"]):
                held_t, stats_t = _read(os.path.join(pc.pool_dir(config, s["records"][t]), "meta.pkl"))
                injected.append((t, pc._assemble_tile(s["inputs"][t], held_t, stats_t, blocks_of[t], flux_grid)))
            band_dict = pc._band_dict(s, injected, flux_grid)
        _atomic_write(os.path.join(sd, "held_%s.pkl" % band), band_dict["held"])
        _atomic_write(path, band_dict)
    if not s.get("empty"):
        for t in s["inputs"]:
            shutil.rmtree(pc.pool_dir(config, s["records"][t]), ignore_errors=True)
        pc._POOL_TABLE.clear()


def finish(config, region, box=None, reference_product=None, n_jobs=None, stage=None, clear=True):
    """Stage FINISH: the region's product, its merged PACS sources and the source table, from the bands' `band.pkl`;
    the stage store is removed after (the held-out halves stay for `acceptance`). `reference_product` is for the
    photometry's checks (not used yet). Returns the bands' datasets."""
    n_jobs = n_jobs or config.n_jobs
    mag, flux_grid = pc.canonical_grid()
    bands = {}
    for b in pc.BANDS:
        path = os.path.join(_band_dir(config, region, b, box), "band.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError("%s: stage assemble of %s %s has not run" % (path, region, b))
        bands[b] = _read(path)
    pc.write_product(product_path(config, "sky/derived", "pacs", "completeness", "source", region=region), region, mag, bands)
    merged = pc.merge_sources(config, bands, n_jobs, stage, flux_grid)
    pc.write_sources(product_path(config, "sky/derived", "pacs", "sources", "region", region=region), region, bands, merged)
    if clear:                  # the command line clears it only after the acceptances and photometry (a rerun resumes)
        _clear_store(config, region, box)
    return bands


def run_region(config, region, box=None, n_shards=1, detect_only=False, stage=None, n_jobs=None):
    """All the stages of a region in order, in this process: a rerun after a kill resumes at the first file missing.
    `n_shards`: the shards each round and the pooling are cut into (the product does not depend on it). `detect_only`:
    the PACS source detection alone (what the merge needs), nothing stored or injected, no completeness product
    written. Returns the bands' datasets."""
    n_jobs = n_jobs or config.n_jobs
    records = prepare(config, region, box, n_jobs, stage)
    mag, flux_grid = pc.canonical_grid()
    if detect_only:
        ra, dec = pc._region_inputs(config, region)
        bands = {}
        for b in pc.BANDS:
            bands[b] = pc._band_detect(config, [records[b]], ra, dec, n_jobs, stage, flux_grid, True) if b in records else None
            if bands[b] is None:
                bands[b] = pc._empty_band(ra.size, mag.size)
        merged = pc.merge_sources(config, bands, n_jobs, stage, flux_grid)
        pc.write_sources(product_path(config, "sky/derived", "pacs", "sources", "region", region=region), region, bands, merged)
        return bands
    for b in pc.BANDS:
        detect(config, region, b, box, n_jobs, stage)
        for r in range(pc.MAX_EXTRA_BATCHES + 1):
            tasks = round_tasks(config, region, b, r, box)
            if not tasks:
                break
            for k in range(n_shards):
                inject(config, region, b, r, k, n_shards, box, n_jobs, stage, tasks=tasks)
        for k in range(n_shards):
            pool(config, region, b, k, n_shards, box, n_jobs, stage)
        assemble(config, region, b, box)
    return finish(config, region, box, n_jobs=n_jobs, stage=stage)


if __name__ == "__main__":
    from sesnaimpute.sky.derived.pacs_stages_cli import main
    raise SystemExit(main())
