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

# Usage: RUNBOOK.sh [--from <module>] [--regions R1 R2 ...]
#   --from     start at the line whose module is <module> (e.g. prior.yso) and run
#              everything after it: a change to one stage rebuilds only its
#              dependants. Lines before it are skipped.
#   --regions  passed to every build; per-region products are rebuilt for those
#              regions only, region-axis tables update those rows in place.
FROM=""; REGIONS=()
while [ $# -gt 0 ]; do case "$1" in
  --from) FROM="$2"; shift 2 ;;
  --regions) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do REGIONS+=("$1"); shift; done ;;
  *) echo "RUNBOOK.sh: unknown argument $1" >&2; exit 2 ;;
esac; done
PYBIN=/usr/local/bin/python3.9
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/src"
CONFIG=/Users/jtaylor/Dropbox/Research/SESNA_Complete/config/root.cfg
STARTED=0; [ -z "$FROM" ] && STARTED=1
# Every stage runs through capped.sh (CODING_RULES 10a) with the region list.
PY() { local module="$1"; shift
  if [ $STARTED -eq 0 ]; then [ "$module" = "sesnaimpute.$FROM" ] && STARTED=1 || return 0; fi
  echo "== $module ${REGIONS[*]:-}"
  "$(dirname "$0")/capped.sh" "$PYBIN" -m "$module" "$CONFIG" ${REGIONS[@]+--regions "${REGIONS[@]}"}
}

# --- downloads ---
# sky/download/<source>/ -- external bytes, verbatim. Download never computes.
PY sesnaimpute.sky.download.hunt_reffert2023.build
PY sesnaimpute.sky.download.baraffe2015_bhac15.build
PY sesnaimpute.sky.download.riebel2012.build
PY sesnaimpute.sky.download.swire.build
PY sesnaimpute.sky.download.scosmos.build
PY sesnaimpute.sky.download.planck_r120.build
PY sesnaimpute.sky.download.herschel_hgbs.build
PY sesnaimpute.sky.download.edenhofer2023.build
# fazio2004 (Fazio et al. 2004, ApJS 154, 39, Table 1, IRAC galaxy source
# counts): no CDS/VizieR entry and IOPscience's machine-readable-table
# service serves no bytes for this DOI (10.1086/422843) -- acquire by hand
# and place verbatim at sky/download/fazio2004/.
PY sesnaimpute.sky.download.trilegal.build
PY sesnaimpute.sky.download.extinction_laws.build
PY sesnaimpute.sky.download.ashby2013_seds.build
PY sesnaimpute.sky.download.h2_knot_surveys.build
# --- catalog ---
# catalog/ -- reads only SESNA: curated catalogues, survey depths, the
# sigma model.
PY sesnaimpute.catalog.curated
PY sesnaimpute.catalog.depths
# limits.py has no build: catalog.limits.limits(config, region) reads curated + depths.
# Downloads keyed on SESNA positions run once the catalogue exists:
PY sesnaimpute.sky.download.gaia_crossmatch.build

# --- sky derived ---
# sky/derived/<source>/ -- one product per computation from those bytes.
# Derive never fetches. Reads SESNA positions and region membership from
# the curated catalogue above, never SESNA fluxes.
PY sesnaimpute.sky.derived.coverage   # Spitzer coverage before granulation (pixel admission)
PY sesnaimpute.sky.derived.knots   # H2S knot survey tables and the Giannini knot-colour ratio distribution (SPEC_PRIORS.md sec 7)
PY sesnaimpute.granules.build
# gaia_counts / twomass_counts (SPEC_PRIORS.md section 2.1, STAR anchor
# counts) build each query from the granule map's own per-region nside-512
# pixel set, so -- like gaia_crossmatch above -- these two download lines
# sit here, after their dependency, rather than in the '--- downloads ---'
# block.
PY sesnaimpute.sky.download.gaia_counts.build
PY sesnaimpute.sky.download.twomass_counts.build
PY sesnaimpute.sky.derived.gaia_counts
PY sesnaimpute.sky.derived.twomass_counts
PY sesnaimpute.sky.derived.gaia_match
PY sesnaimpute.sky.derived.herschel_column
PY sesnaimpute.sky.derived.subbeam
PY sesnaimpute.sky.derived.planck_column
PY sesnaimpute.sky.derived.planck_source_column
PY sesnaimpute.sky.derived.column
PY sesnaimpute.sky.derived.profile

# --- prior ---
# bms/ prior products: the column grid (every class build reads its
# nodes), the column kernel, the PAHC curve, depth groups, field stars,
# anchor weights, the per-class priors and selection tables, the prior
# table.
PY sesnaimpute.prior.column_grid
PY sesnaimpute.prior.field_stars
PY sesnaimpute.prior.pahc_curve   # PAHC contamination P(q), survey-wide (SPEC_PRIORS.md sec 4)
PY sesnaimpute.prior.anchor_tiles
PY sesnaimpute.prior.yso   # YSO law count + per-sightline shape (SPEC_PRIORS.md sec 6.1, 6.3)
PY sesnaimpute.prior.young_stars   # expected young stars per STAR anchor pixel/bin (SPEC_PRIORS.md sec 2.1)
PY sesnaimpute.prior.anchor_observed   # observed STAR anchor joint histogram, young-star subtracted (SPEC_PRIORS.md sec 2.1)
PY sesnaimpute.prior.anchor_weights   # per-tile STAR anchor weight W, cluster exclusion, faint-end trend (SPEC_PRIORS.md sec 2.1)
PY sesnaimpute.prior.star_population   # per-tile field-star placement (u, a) and anchor weight W (SPEC_PRIORS.md sec 1.4, 1.5, 2.1, 2.2)
PY sesnaimpute.prior.star_shapes   # per-tile STAR/AGB/PAHC shapes, kernel-convolved at fidelity-chosen nodes and grid (SPEC_PRIORS.md sec 2.2, 3, 4; IMPLEMENTATION.md sec 3)
PY sesnaimpute.prior.star_selection   # STAR/AGB/PAHC exact selection per source (SPEC_PRIORS.md sec 1.3)
PY sesnaimpute.prior.yso_selection
PY sesnaimpute.prior.h2s                    # H2S: law-blurred field, region lognormal + knot-colour selection (SPEC_PRIORS.md sec 7)
PY sesnaimpute.prior.gal                    # GAL: Fazio counts law + SWIRE selection tables
PY sesnaimpute.prior.counts_star_family
PY sesnaimpute.prior.counts_cloud           # YSO/H2S per-source counts on the cloud law (SPEC_PRIORS.md sec 6.2, 7)
PY sesnaimpute.prior.levels                 # the one scalar per region normalising the six counts to the catalogued source count (SPEC_PRIORS.md sec 0.2)
PY sesnaimpute.prior.table                  # the prior table: the join, one row per catalogue source (IMPLEMENTATION.md sec 5)

# --- fit ---
# bms/ posterior fit of class and subclass probabilities from the prior
# table.
PY sesnaimpute.fit.run                      # the class-posterior evidence sweep, batched (IMPLEMENTATION.md sec 5; 10_POSTERIOR.md sec 1)

# --- impute ---
# bms/ final impute decisions and diagnostics.
PY sesnaimpute.impute.posterior             # class/subclass posteriors, argmax-commit flux imputation, entropies (10_POSTERIOR.md sec 1)
