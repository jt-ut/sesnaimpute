#!/usr/bin/env bash
# RUNBOOK.sh -- the only orchestration for sesnaimpute.
#
# This file's line order IS the dependency order: each stage below reads
# only the products of stages already listed above it. Each line
# overwrites its own product. A manual, one-off acquisition appears as a
# comment line carrying the exact URL and destination, not as a script
# step. A build whose input file is missing fails with one sentence
# naming the RUNBOOK line that makes it -- that is the only existence
# check in this pipeline.
set -euo pipefail

PY=/usr/local/bin/python3.9
CONFIG=/Users/jtaylor/Dropbox/Research/SESNA_Complete/config/root.cfg

# --- downloads ---
# sky/download/<source>/ -- external bytes, verbatim. Download never computes.
$PY -m sesnaimpute.sky.download.hunt_reffert2023.build $CONFIG
$PY -m sesnaimpute.sky.download.baraffe2015_bhac15.build $CONFIG
$PY -m sesnaimpute.sky.download.riebel2012.build $CONFIG
$PY -m sesnaimpute.sky.download.swire.build $CONFIG
$PY -m sesnaimpute.sky.download.scosmos.build $CONFIG
$PY -m sesnaimpute.sky.download.planck_r120.build $CONFIG
$PY -m sesnaimpute.sky.download.herschel_hgbs.build $CONFIG
$PY -m sesnaimpute.sky.download.edenhofer2023.build $CONFIG
# fazio2004 (Fazio et al. 2004, ApJS 154, 39, Table 1, IRAC galaxy source
# counts): no CDS/VizieR entry and IOPscience's machine-readable-table
# service serves no bytes for this DOI (10.1086/422843) -- acquire by hand
# and place verbatim at sky/download/fazio2004/.
$PY -m sesnaimpute.sky.download.trilegal.build $CONFIG
$PY -m sesnaimpute.sky.download.extinction_laws.build $CONFIG
$PY -m sesnaimpute.sky.download.ashby2013_seds.build $CONFIG
$PY -m sesnaimpute.sky.download.h2_knot_surveys.build $CONFIG
# --- catalog ---
# catalog/ -- reads only SESNA: curated catalogues, survey depths, the
# sigma model.
$PY -m sesnaimpute.catalog.curated $CONFIG
$PY -m sesnaimpute.catalog.depths $CONFIG
# limits.py has no build: catalog.limits.limits(config, region) reads curated + depths.
# Downloads keyed on SESNA positions run once the catalogue exists:
$PY -m sesnaimpute.sky.download.gaia_crossmatch.build $CONFIG

# --- sky derived ---
# sky/derived/<source>/ -- one product per computation from those bytes.
# Derive never fetches. Reads SESNA positions and region membership from
# the curated catalogue above, never SESNA fluxes.
$PY -m sesnaimpute.sky.derived.coverage $CONFIG   # Spitzer coverage before granulation (pixel admission)
$PY -m sesnaimpute.sky.derived.knots $CONFIG   # H2S knot survey tables and the Giannini knot-colour ratio distribution (SPEC_PRIORS.md sec 7)
$PY -m sesnaimpute.granules.build $CONFIG
# gaia_counts / twomass_counts (SPEC_PRIORS.md section 2.1, STAR anchor
# counts) build each query from the granule map's own per-region nside-512
# pixel set, so -- like gaia_crossmatch above -- these two download lines
# sit here, after their dependency, rather than in the '--- downloads ---'
# block.
$PY -m sesnaimpute.sky.download.gaia_counts.build $CONFIG
$PY -m sesnaimpute.sky.download.twomass_counts.build $CONFIG
$PY -m sesnaimpute.sky.derived.gaia_counts $CONFIG
$PY -m sesnaimpute.sky.derived.twomass_counts $CONFIG
$PY -m sesnaimpute.sky.derived.gaia_match $CONFIG
$PY -m sesnaimpute.sky.derived.herschel_column $CONFIG
$PY -m sesnaimpute.sky.derived.subbeam $CONFIG
$PY -m sesnaimpute.sky.derived.planck_column $CONFIG
$PY -m sesnaimpute.sky.derived.planck_source_column $CONFIG
$PY -m sesnaimpute.sky.derived.column $CONFIG
$PY -m sesnaimpute.sky.derived.profile $CONFIG

# --- prior ---
# bms/ prior products: the column grid (every class build reads its
# nodes), the column kernel, the PAHC curve, depth groups, field stars,
# anchor weights, the per-class priors and selection tables, the prior
# table.
$PY -m sesnaimpute.prior.column_grid $CONFIG
$PY -m sesnaimpute.prior.field_stars $CONFIG
$PY -m sesnaimpute.prior.pahc_curve $CONFIG   # PAHC contamination P(q), survey-wide (SPEC_PRIORS.md sec 4)
$PY -m sesnaimpute.prior.anchor_tiles $CONFIG
$PY -m sesnaimpute.prior.depth_groups $CONFIG
$PY -m sesnaimpute.prior.yso $CONFIG   # YSO law count + per-sightline shape (SPEC_PRIORS.md sec 6.1, 6.3)
$PY -m sesnaimpute.prior.young_stars $CONFIG   # expected young stars per STAR anchor pixel/bin (SPEC_PRIORS.md sec 2.1)
$PY -m sesnaimpute.prior.anchor_observed $CONFIG   # observed STAR anchor joint histogram, young-star subtracted (SPEC_PRIORS.md sec 2.1)
$PY -m sesnaimpute.prior.anchor_weights $CONFIG   # per-tile STAR anchor weight W, cluster exclusion, faint-end trend (SPEC_PRIORS.md sec 2.1)
$PY -m sesnaimpute.prior.star_population $CONFIG   # per-tile field-star placement (u, a) and anchor weight W (SPEC_PRIORS.md sec 1.4, 1.5, 2.1, 2.2)
$PY -m sesnaimpute.prior.star_shapes $CONFIG   # per-tile STAR/AGB/PAHC shapes, kernel-convolved at fidelity-chosen nodes and grid (SPEC_PRIORS.md sec 2.2, 3, 4; IMPLEMENTATION.md sec 3)
$PY -m sesnaimpute.prior.star_selection $CONFIG   # STAR/AGB/PAHC exact-selection tables per depth group (SPEC_PRIORS.md sec 1.3, 2.1, 3, 4)
$PY -m sesnaimpute.prior.yso_selection $CONFIG
$PY -m sesnaimpute.prior.h2s $CONFIG                    # H2S: law-blurred field, region lognormal + knot-colour selection (SPEC_PRIORS.md sec 7)
$PY -m sesnaimpute.prior.gal $CONFIG                    # GAL: Fazio counts law + SWIRE selection tables
$PY -m sesnaimpute.prior.counts_star_family $CONFIG
$PY -m sesnaimpute.prior.counts_cloud $CONFIG           # YSO/H2S per-source counts on the cloud law (SPEC_PRIORS.md sec 6.2, 7)

# --- fit ---
# bms/ posterior fit of class and subclass probabilities from the prior
# table.

# --- impute ---
# bms/ final impute decisions and diagnostics.
