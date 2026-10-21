"""Uploads each region's SESNA positions to the Gaia archive and pulls every
Gaia DR3 candidate within the query radius. Download never computes: no
proper-motion propagation to the survey epoch, no nearest-candidate
selection, no match probability -- that is SPEC_PRIORS.md section 2.1's
separate match-probability build, not this module.

SERVICE: CDS X-Match (astroquery.xmatch.XMatch), anonymous, uploading
NAME/RA_DEG/DEC_DEG (the curated catalogue's own columns) as cat1 against
cat2 `vizier:I/355/gaiadr3` (Gaia DR3), with no `selection` override, which
is the service's default of returning every candidate pair in the radius,
not just the nearest one:

    XMatch.query(cat1=<NAME, RA_DEG, DEC_DEG for one chunk>,
                 cat2="vizier:I/355/gaiadr3",
                 max_distance=QUERY_RADIUS_ARCSEC * u.arcsec,
                 colRA1="RA_DEG", colDec1="DEC_DEG",
                 cols2=<CAT2_COLUMNS, comma-joined>)

RADIUS: 3.0 arcsec. This is a net, not an acceptance radius: Gaia DR3's
astrometry is given at epoch 2016.0 while SESNA was surveyed around a
decade earlier, so a real counterpart can sit outside a tight cone purely
from its own proper motion by the time of the SESNA epoch; the wider net
catches that swing so a later, epoch-aware selection (propagate each
candidate's own proper motion back to the survey epoch, then pick the
nearest) has every true counterpart to choose from. This module ships
every candidate the net returns; it does not choose among them.

COLUMNS: `CAT2_COLUMNS` below is the old sesnacomplete `gaia_source_crossmatch`
module's field list, read from `region_pull.VIZIER_RENAME` (source_id,
position, parallax and its error, both proper motions and their errors,
the RUWE astrometric-quality flag, and the G/BP/RP mean magnitudes) --
restricting the response to this list, rather than the service's ~141
default columns, is also what the old module found fixed an upstream
VOTable serialization failure on wide responses. `angDist` is appended by
the service itself. Written to disk is the response exactly as returned,
plus the SESNA `NAME`, `RA_DEG`, `DEC_DEG` the service already echoes back
per candidate row since they were uploaded as cat1 columns.

CHUNKING: the old client's per-request row cap on the *uploaded* source
table (`xmatch_client.CHUNK_SIZE_DEFAULT`/`CHUNK_SIZE_DENSE`): 500,000
sources per request, halved to 250,000 for Cygnus X, dense enough (about
38% of the SESNA catalogue) to have risked the service's 2,000,000-row
result cap at typical crowding. Chunks for one region are queried strictly
serially, with a politeness delay between requests, since CDS's own
concurrency tolerance is undocumented and an institution-wide IP ban is
the failure mode of exceeding it.
"""

import os
import time

import astropy.units as u
import h5py
import numpy as np
import pandas as pd
from astropy.table import Table
from astroquery.xmatch import XMatch

from sesnaimpute import config as config_module
from sesnaimpute import progress as progress_module
from sesnaimpute.build import run
from sesnaimpute.regions import REGIONS

XMATCH_CATALOG = "vizier:I/355/gaiadr3"
QUERY_RADIUS_ARCSEC = 3.0

# Source rows uploaded per request; halved for the one region dense enough
# to risk the service's result-row cap (old xmatch_client.py).
CHUNK_SIZE_DEFAULT = 500_000
CHUNK_SIZE_DENSE = 250_000
DENSE_REGIONS = frozenset({"Cygnus X"})

# Delay between serial chunk requests to the same anonymous service.
POLITE_DELAY_S = 3.0

# vizier:I/355/gaiadr3 fields requested via cols2, verbatim VizieR spellings
# (old sesnacomplete.fetch_external.gaia_source_crossmatch.region_pull.VIZIER_RENAME
# keys, minus angDist which the service appends on its own).
CAT2_COLUMNS = (
    "Source", "RAdeg", "DEdeg", "Plx", "e_Plx",
    "pmRA", "e_pmRA", "pmDE", "e_pmDE", "RUWE",
    "Gmag", "BPmag", "RPmag",
)


def _chunk_size(region):
    return CHUNK_SIZE_DENSE if region in DENSE_REGIONS else CHUNK_SIZE_DEFAULT


def _read_positions(config, region):
    """NAME, RA_DEG, DEC_DEG from the curated catalogue -- SESNA's own
    positions, never SESNA's measured fluxes (IMPLEMENTATION.md section 1a)."""
    path = config_module.product_path(
        config, "catalog", "sesna", "sources", "source", region=region)
    with h5py.File(path, "r") as f:
        name = np.array([
            s.decode() if isinstance(s, bytes) else s for s in f["NAME"][()]])
        ra_deg = f["RA_DEG"][()]
        dec_deg = f["DEC_DEG"][()]
    return pd.DataFrame({"NAME": name, "RA_DEG": ra_deg, "DEC_DEG": dec_deg})


def _query_chunk(chunk_df):
    """One CDS X-Match request for one chunk of SESNA positions; returns
    every candidate pair exactly as the service reports it."""
    cat1 = Table.from_pandas(chunk_df[["NAME", "RA_DEG", "DEC_DEG"]])
    response = XMatch.query(
        cat1=cat1, cat2=XMATCH_CATALOG,
        max_distance=QUERY_RADIUS_ARCSEC * u.arcsec,
        colRA1="RA_DEG", colDec1="DEC_DEG",
        cols2=",".join(CAT2_COLUMNS),
    )
    return response.to_pandas()


def _pull_region(positions, region):
    """Chunks `positions` at the region's upload row cap and queries each
    chunk serially, concatenating every returned candidate row."""
    size = _chunk_size(region)
    n = len(positions)
    n_chunks = max(1, -(-n // size))
    frames = []
    for i in range(n_chunks):
        lo, hi = i * size, min((i + 1) * size, n)
        frames.append(_query_chunk(positions.iloc[lo:hi]))
        if i < n_chunks - 1:
            time.sleep(POLITE_DELAY_S)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def uploaded_names_path(config, region):
    """The record of the SESNA NAMEs uploaded for `region`'s candidates file,
    one per line, beside it: the candidates' own definition, read by
    `build` (skip only when it equals the catalogue) and by
    `sky.derived.gaia_match` (every catalogue source must have been
    uploaded)."""
    return f"{config.data_root}/sky/download/gaia_crossmatch/uploaded_gaia_source__{region}.txt"


def read_uploaded_names(path):
    with open(path) as fh:
        return set(line.rstrip("\n") for line in fh if line.strip())


def _stale_reason(dest_path, upload_path, names_now):
    """None when the candidates file was uploaded from the catalogue now on
    disk; else a sentence saying how it differs (reported, never acted on)."""
    if not os.path.exists(dest_path):
        return "no candidates file"
    if not os.path.exists(upload_path):
        return "no record of the sources it was uploaded from (pulled before 2026-10-22)"
    uploaded = read_uploaded_names(upload_path)
    now = set(names_now.tolist())
    if uploaded == now:
        return None
    return (f"the curated catalogue changed since the pull: {len(now - uploaded)} source(s) "
            f"never uploaded, {len(uploaded - now)} uploaded source(s) no longer in the catalogue")


def build(config, regions=None):
    """Uploads each region's SESNA positions to Gaia DR3 X-Match and writes
    every returned candidate, verbatim, to
    `sky/download/gaia_crossmatch/candidates_gaia_source__<Region>.csv`."""
    if regions is None:
        regions = [r.name for r in REGIONS]
    dest_dir = f"{config.data_root}/sky/download/gaia_crossmatch"
    os.makedirs(dest_dir, exist_ok=True)
    with progress_module.Stage("sky.download.gaia_crossmatch") as st:
        n_regions = len(regions)
        for i, region in enumerate(regions):
            dest_path = f"{dest_dir}/candidates_gaia_source__{region}.csv"
            upload_path = uploaded_names_path(config, region)
            positions = _read_positions(config, region)
            # A present candidates file is used as it is (owner ruling
            # 2026-10-22: the data on disk is not re-queried). The upload
            # record beside it, where one exists, only REPORTS how the curated
            # catalogue differs from the uploaded list; it never triggers a
            # pull. A missing file is pulled and its record written.
            if os.path.exists(dest_path):
                note = _stale_reason(dest_path, upload_path, positions["NAME"])
                print(f"gaia_crossmatch build: {dest_path} present, skipped"
                      + (f" ({note})" if note else " (uploaded from the current catalogue)"))
            else:
                print(f"gaia_crossmatch build: {region}: no candidates file; pulling")
                candidates = _pull_region(positions, region)
                candidates.to_csv(dest_path, index=False)
                with open(upload_path, "w") as fh:
                    fh.write("\n".join(positions["NAME"].tolist()) + "\n")
                n_sources = len(positions)
                n_candidates = len(candidates)
                per_source = n_candidates / n_sources if n_sources else float("nan")
                print(f"gaia_crossmatch build: {region} -> {dest_path}: "
                      f"{n_sources} sources uploaded, {n_candidates} candidates "
                      f"returned ({per_source:.3f} candidates/source)")
            st.tick(i + 1, n_regions, "regions")
        st.done(dest_dir, regions=n_regions)


if __name__ == "__main__":
    run(build)
