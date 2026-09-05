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
$PY -m sesnaimpute.sky.download.fazio2004.build $CONFIG
$PY -m sesnaimpute.sky.download.trilegal.build $CONFIG
$PY -m sesnaimpute.sky.download.extinction_laws.build $CONFIG
# ashby2013_seds (VizieR J/ApJ/769/80, SEDS Tables 7-11 + ReadMe): no
# programmatic CDS/VizieR URL was ever pinned down for these five tables;
# acquire by hand and place verbatim at sky/download/ashby2013_seds/.
# h2_knot_surveys: Giannini+2013 (VizieR J/ApJ/767/147), Davis+2009
# (VizieR J/A+A/496/153), Walawender+2005 (AJ 129/130), UWISH2 Table D1
# (Froebrich+2015, MNRAS 454, 2586, behind institutional auth) -- no
# reachable URL for any of the four; acquire by hand and place verbatim
# at sky/download/h2_knot_surveys/.
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
$PY -m sesnaimpute.prior.anchor_tiles $CONFIG
$PY -m sesnaimpute.prior.depth_groups $CONFIG
$PY -m sesnaimpute.prior.yso $CONFIG   # YSO law count + per-sightline shape (SPEC_PRIORS.md sec 6.1, 6.3)

# --- fit ---
# bms/ posterior fit of class and subclass probabilities from the prior
# table.

# --- impute ---
# bms/ final impute decisions and diagnostics.
