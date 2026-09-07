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
  *) echo "RUNBOOK.sh: unknown argument $1" >&2; exit 2 ;;
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
PY sesnaimpute.sky.derived.swire_galaxies   # SWIRE galaxies after the adopted star-galaxy split, survey-wide (SPEC_BMSTP_DRAFT.md sec 3.2, 5.4)

# --- prior ---
# bms/ prior products: the column grid (every class build reads its
# nodes), the column kernel, the PAHC curve, depth groups, field stars,
# anchor weights, the per-class priors and selection tables, the prior
# table.
PY sesnaimpute.prior.column_grid
PY sesnaimpute.prior.field_stars
PY sesnaimpute.prior.pahc_curve   # PAHC contamination P(q), survey-wide (SPEC_PRIORS.md sec 4)
PY sesnaimpute.prior.anchor_tiles
PY sesnaimpute.prior.kernel   # the log-normal column kernel: mu, sigma per arm on the column grid (SPEC_PRIORS.md sec 1.2)
PY sesnaimpute.prior.yso   # YSO law count + per-sightline shape (SPEC_PRIORS.md sec 6.1, 6.3)
PY sesnaimpute.prior.young_stars   # expected young stars per STAR anchor pixel/bin (SPEC_PRIORS.md sec 2.1)
PY sesnaimpute.prior.anchor_observed   # observed STAR anchor joint histogram, young-star subtracted (SPEC_PRIORS.md sec 2.1)
PY sesnaimpute.prior.anchor_weights   # per-tile STAR anchor weight W, cluster exclusion, faint-end trend (SPEC_PRIORS.md sec 2.1)
PY sesnaimpute.prior.star_population   # per-tile field-star placement (u, a) and anchor weight W (SPEC_PRIORS.md sec 1.4, 1.5, 2.1, 2.2)
PY sesnaimpute.prior.star_shapes   # per-tile STAR/AGB/PAHC shapes on a fixed 64x64 grid and fixed shape nodes, log-normal kernel shift+smoothing (SPEC_PRIORS.md sec 2.2, 3, 4; IMPLEMENTATION.md sec 3)
PY sesnaimpute.prior.star_selection   # STAR/AGB/PAHC exact selection per source (SPEC_PRIORS.md sec 1.3)
PY sesnaimpute.prior.yso_selection   # YSO IMF-mass selection, exact per source on X_LADDER (SPEC_PRIORS.md sec 1.3, 6.2)
PY sesnaimpute.prior.h2s                    # H2S: law-blurred field, region lognormal, exact per-source knot-colour selection (SPEC_PRIORS.md sec 1.3, 7)
PY sesnaimpute.prior.gal                    # GAL: Fazio counts law + survey-wide SWIRE colour-CDF tables, selection read on the fly, no per-source product (SPEC_PRIORS.md sec 1.3, 5)
PY sesnaimpute.prior.counts_star_family     # STAR/AGB/PAHC/GAL per-source counts and normalisers, exact selection against the shape (SPEC_PRIORS.md sec 0.2, 2-5)
PY sesnaimpute.prior.counts_cloud           # YSO/H2S per-source counts on the cloud law (SPEC_PRIORS.md sec 6.2, 7)
PY sesnaimpute.prior.levels                 # the one scalar per region normalising the six counts to the catalogued source count (SPEC_PRIORS.md sec 0.2)
PY sesnaimpute.prior.table                  # the prior table: the join, one row per catalogue source (IMPLEMENTATION.md sec 5)

# --- fit ---
# bms/ posterior fit of class and subclass probabilities from the prior
# table. One job is one {region, class} (fit.run's own module docstring)
# and the NGC 7129 dry run (owner ruling 2026-09-06) found resident
# memory is NOT released between classes swept in one process (STAR 4.0
# GB -> AGB 6.2 -> PAHC 6.7 -> GAL 7.8 -> YSO 7.8 -> H2S killed above the
# 8 GB capped.sh ceiling) -- so this line runs the six classes as six
# SEPARATE `capped.sh`-wrapped processes, in `fit.run.CLASSES`'s own
# order (STAR AGB PAHC GAL YSO H2S), each capped and reported on its own:
# one class's peak is then that class's process's own peak, never summed
# with the class before it. `fit.run.build`'s in-process all-six-classes
# path (its own `--classes` default) still exists for another caller --
# e.g. a cluster's `bms/fit/jobs.sh` (below), already one job per line --
# but this RUNBOOK always passes one `--classes` value, so that
# in-process multi-class path never actually runs here.
for FIT_CLASS in STAR AGB PAHC GAL YSO H2S; do
  PY sesnaimpute.fit.run --classes "$FIT_CLASS"   # the class-posterior evidence sweep, batched (IMPLEMENTATION.md sec 5; 10_POSTERIOR.md sec 1)
done
# bms/fit/jobs.sh (one line per {region, class} job, for a cluster) is
# not built by this RUNBOOK -- it targets a different machine than the
# one running this script. Write/refresh it by hand with:
#   $PYBIN -m sesnaimpute.fit.run $CONFIG --jobs

# --- impute ---
# bms/ final impute decisions and diagnostics.
PY sesnaimpute.impute.posterior             # class/subclass posteriors, argmax-commit flux imputation, entropies (10_POSTERIOR.md sec 1)

# A --from that never matched any PY line above would otherwise leave
# every stage silently skipped and the script exiting 0 having done
# nothing -- fail loudly instead.
if [ -n "$FROM" ] && [ $STARTED -eq 0 ]; then
  echo "RUNBOOK.sh: --from $FROM matched no stage" >&2
  exit 2
fi
