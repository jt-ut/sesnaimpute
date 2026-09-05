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
# ashby2013_seds (VizieR J/ApJ/769/80, SEDS Tables 7-11 + ReadMe): no
# programmatic CDS/VizieR URL was ever pinned down for these five tables;
# acquire by hand and place verbatim at sky/download/ashby2013_seds/.
# h2_knot_surveys: Giannini+2013 (VizieR J/ApJ/767/147), Davis+2009
# (VizieR J/A+A/496/153), Walawender+2005 (AJ 129/130), UWISH2 Table D1
# (Froebrich+2015, MNRAS 454, 2586, behind institutional auth) -- no
# reachable URL for any of the four; acquire by hand and place verbatim
# at sky/download/h2_knot_surveys/.

# --- sky derived ---
# sky/derived/<source>/ -- one product per computation from those bytes.
# Derive never fetches.

$PY -m sesnaimpute.granules.build $CONFIG
$PY -m sesnaimpute.sky.derived.subbeam $CONFIG
$PY -m sesnaimpute.sky.derived.planck_column $CONFIG

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
$PY -m sesnaimpute.prior.depth_groups $CONFIG

# --- fit ---
# bms/ posterior fit of class and subclass probabilities from the prior
# table.

# --- impute ---
# bms/ final impute decisions and diagnostics.
