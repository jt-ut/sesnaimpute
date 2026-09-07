#!/usr/bin/env bash
# RUNBOOKtp.sh -- the orchestration for the thinned-Poisson design: the inputs,
# the population summaries, the bmstp prior, the fittp fitter and posterior.
# RUNBOOK.sh drives the earlier design; the input stages are shared verbatim.
#
# This file's line order IS the dependency order: each stage below reads
# only the products of stages already listed above it. Each line
# overwrites its own product. A manual, one-off acquisition appears as a
# comment line carrying the exact URL and destination, not as a script
# step. A build whose input file is missing fails with one sentence
# naming the RUNBOOK line that makes it -- that is the only existence
# check in this pipeline.
set -euo pipefail

# Usage: RUNBOOKtp.sh [--from <module>] [--regions R1 R2 ...]
#   --from     start at the line whose module is <module> (fully qualified,
#              e.g. sesnaimpute.prior.yso, matching the PY lines below
#              verbatim) and run everything after it: a change to one stage
#              rebuilds only its dependants. Lines before it are skipped.
#   --regions  passed to every build; per-region products are rebuilt for those
#              regions only, region-axis tables update those rows in place.
FROM=""; REGIONS=()
while [ $# -gt 0 ]; do case "$1" in
  --from) FROM="$2"; shift 2 ;;
  --regions) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do REGIONS+=("$1"); shift; done ;;
  *) echo "RUNBOOKtp.sh: unknown argument $1" >&2; exit 2 ;;
esac; done
PYBIN=/usr/local/bin/python3.9
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/src"
CONFIG=/Users/jtaylor/Dropbox/Research/SESNA_Complete/config/root.cfg
STARTED=0; [ -z "$FROM" ] && STARTED=1
# Every stage runs through capped.sh (CODING_RULES 10a) with the region list.
# Extra arguments after the module name (e.g. the fit loop's own
# `--classes <C>`, below) pass straight through to that one invocation --
# capped.sh then caps that one process alone, so a line that calls PY
# more than once (one process per call) never shares one process's
# memory ceiling with another.
PY() { local module="$1"; shift
  if [ $STARTED -eq 0 ]; then [ "$module" = "$FROM" ] && STARTED=1 || return 0; fi
  echo "== $module ${REGIONS[*]:-} $*"
  "$(dirname "$0")/capped.sh" "$PYBIN" -m "$module" "$CONFIG" ${REGIONS[@]+--regions "${REGIONS[@]}"} "$@"
}

# --- downloads ---
# sky/download/<source>/ -- external bytes, verbatim. Download never computes.
PY sesnaimpute.sky.download.hunt_reffert2023.build
PY sesnaimpute.sky.download.baraffe2015_bhac15.build
PY sesnaimpute.sky.download.mist2016.build   # MIST v1.2 1 Myr isochrone, the mass-luminosity relation above BHAC15's 1.4 Msun top (SPEC_BMSTP sec 3.5)
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
PY sesnaimpute.sky.download.h2_knot_surveys.build
# --- catalog ---
# catalog/ -- reads only SESNA: curated catalogues, survey depths, the
# sigma model.
PY sesnaimpute.catalog.curated
PY sesnaimpute.catalog.depths
PY sesnaimpute.catalog.depth_grid   # median 50 % limits per admitted hpx512 pixel, for the prior atlas (SPEC_BMSTP sec 3.3)
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
PY sesnaimpute.sky.derived.swire_galaxies   # SWIRE galaxies after the adopted star-galaxy split, survey-wide (SPEC_BMSTP sec 3.2, 5.4)


# --- population (the model-driven summaries the bmstp prior ingests; writes population/) ---
# Copies of the earlier design's stages, science unchanged (SPEC_PRIORS.md); only the area differs.
PY sesnaimpute.population.column_grid
PY sesnaimpute.population.field_stars
PY sesnaimpute.population.yso_mass   # each YSO template's stellar mass through the 1 Myr isochrone (SPEC_BMSTP sec 3.5)
PY sesnaimpute.population.pahc_curve
PY sesnaimpute.population.anchor_tiles
PY sesnaimpute.population.kernel
PY sesnaimpute.population.yso
PY sesnaimpute.population.yso_selection
PY sesnaimpute.population.h2s
PY sesnaimpute.population.gal
PY sesnaimpute.population.young_stars
PY sesnaimpute.population.anchor_observed
PY sesnaimpute.population.anchor_weights
PY sesnaimpute.population.star_population

# --- bmstp (the prior: shape grids per grain, the per-source density table, template weights, the atlas; writes bmstp/) ---
# Lines are added here as each stage lands (IMPLEMENTATION_BMSTP sec 3).

# --- fittp (the thinned-Poisson fitter, the classification, the cascade, the posterior atlas; writes fittp/) ---
# Lines are added here as each stage lands (IMPLEMENTATION_BMSTP sec 4). The [fittp] knobs live in root.cfg.
PY sesnaimpute.fittp.cascade   # the colour cascade on the measured fluxes, Psi per source (SPEC_BMSTP sec 6.5)

# A --from that never matched any PY line above would otherwise leave
# every stage silently skipped and the script exiting 0 having done
# nothing -- fail loudly instead.
if [ -n "$FROM" ] && [ $STARTED -eq 0 ]; then
  echo "RUNBOOKtp.sh: --from $FROM matched no stage" >&2
  exit 2
fi
