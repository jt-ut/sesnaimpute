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

# --- sky derived ---
# sky/derived/<source>/ -- one product per computation from those bytes.
# Derive never fetches.

$PY -m sesnaimpute.granules.build $CONFIG
# TODO: move below the planck sightline column build once its RUNBOOK line lands (profile.build reads its product).
$PY -m sesnaimpute.sky.derived.profile $CONFIG   # writes profile_edenhofer_sightline and depth_edenhofer_region together, one build

# --- catalog ---
# catalog/ -- reads only SESNA: curated catalogues, survey depths, the
# sigma model.
$PY -m sesnaimpute.catalog.curated $CONFIG
$PY -m sesnaimpute.catalog.depths $CONFIG
# limits.py has no build: catalog.limits.limits(config, region) reads curated + depths.

# --- prior ---
# bms/ prior products: column kernel, PAHC curve, column grid, depth
# groups, field stars, anchor weights, the per-class priors and
# selection tables, the prior table.

# --- fit ---
# bms/ posterior fit of class and subclass probabilities from the prior
# table.

# --- impute ---
# bms/ final impute decisions and diagnostics.
