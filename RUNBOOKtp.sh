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
#
# Every joblib-parallelised stage's worker count is `root.cfg`'s own
# `[run] n_jobs`, with no separate module-level cap (CODING_RULES_BMSTP.md
# rule 10a): the owner sets it to what the machine's memory allows.
# `NUMBA_NUM_THREADS` governs the numba kernels in fittp's prior reader
# (`fittp.prior_reader`) and its likelihood (`fittp.likelihood`)
# separately from that worker count; the sweep (`fittp.sweep`) pins BLAS
# to one thread for its small per-source gemms while those numba kernels
# keep their own thread pool.
set -euo pipefail

# Usage: RUNBOOKtp.sh [--from <module>] [--to <module>] [--regions R1 R2 ...]
#   --from     start at the line whose module is <module> (fully qualified,
#              e.g. sesnaimpute.prior.yso, matching the PY lines below
#              verbatim) and run everything after it: a change to one stage
#              rebuilds only its dependants. Lines before it are skipped.
#   --to       stop after the line whose module is <module> (inclusive): a prior-only
#              build is `--from sesnaimpute.population.column_grid --to sesnaimpute.bmstp.atlas`.
#   --regions  passed to every build; per-region products are rebuilt for those
#              regions only, region-axis tables update those rows in place.
FROM=""; TO=""; REGIONS=()
while [ $# -gt 0 ]; do case "$1" in
  --from) FROM="$2"; shift 2 ;;
  --to) TO="$2"; shift 2 ;;
  --regions) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do REGIONS+=("$1"); shift; done ;;
  *) echo "RUNBOOKtp.sh: unknown argument $1" >&2; exit 2 ;;
esac; done
PYBIN=/usr/local/bin/python3.9
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/src"
CONFIG=/Users/jtaylor/Dropbox/Research/SESNA_Complete/config/root.cfg
STARTED=0; [ -z "$FROM" ] && STARTED=1
STOPPED=0
# Every stage runs through capped.sh (CODING_RULES 10a) with the region list.
# Extra arguments after the module name (e.g. the fit loop's own
# `--classes <C>`, below) pass straight through to that one invocation --
# capped.sh then caps that one process alone, so a line that calls PY
# more than once (one process per call) never shares one process's
# memory ceiling with another.
PY() { local module="$1"; shift
  if [ $STARTED -eq 0 ]; then [ "$module" = "$FROM" ] && STARTED=1 || return 0; fi
  [ $STOPPED -eq 1 ] && return 0
  echo "== $module ${REGIONS[*]:-} $*"
  "$(dirname "$0")/capped.sh" "$PYBIN" -m "$module" "$CONFIG" ${REGIONS[@]+--regions "${REGIONS[@]}"} "$@"
  [ -n "$TO" ] && [ "$module" = "$TO" ] && STOPPED=1
  return 0
}

# --- downloads ---
# sky/download/<source>/ -- external bytes, verbatim. Download never computes.
PY sesnaimpute.sky.download.hunt_reffert2023.build
PY sesnaimpute.sky.download.hops.build   # HOPS/eHOPS Herschel-confirmed protostars, report-only overlay for the protostar check (SPEC_BMSTP sec 5.5, sec 9)
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
PY sesnaimpute.catalog.depth_grid   # median and marginalised 50 % limits + width per admitted hpx512 pixel, for the prior atlas (SPEC_BMSTP sec 3.3)
PY sesnaimpute.catalog.coverage   # the atlas's coverage: fraction of each admitted hpx512 pixel with a measured IRAC detection, the catalogue's own footprint (SPEC_BMSTP sec 3.3, sec 8)
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
PY sesnaimpute.sky.derived.column   # the adopted column per source and sightline (its survey-wide disagreement check is not run: an (n_source) array over 8.7 M sources, the earlier design's)
PY sesnaimpute.sky.derived.profile
PY sesnaimpute.sky.derived.swire_galaxies   # SWIRE galaxies after the adopted star-galaxy split, survey-wide (SPEC_BMSTP sec 3.2, 5.4)
PY sesnaimpute.sky.derived.protostars   # HOPS + eHOPS pooled in one schema, report-only overlay for the protostar check (SPEC_BMSTP sec 5.5, sec 9)


# --- population (the model-driven summaries the bmstp prior ingests; writes population/) ---
# Copies of the earlier design's stages, science unchanged (SPEC_PRIORS.md); only the area differs.
PY sesnaimpute.population.column_grid
PY sesnaimpute.population.field_stars
PY sesnaimpute.population.yso_mass   # each YSO template's stellar mass through the 1 Myr isochrone (SPEC_BMSTP sec 3.5)
PY sesnaimpute.population.pahc_curve
PY sesnaimpute.population.anchor_tiles
PY sesnaimpute.population.kernel
PY sesnaimpute.population.yso
PY sesnaimpute.population.gal
PY sesnaimpute.population.young_stars
PY sesnaimpute.population.anchor_observed
PY sesnaimpute.population.anchor_weights
PY sesnaimpute.population.star_population
PY sesnaimpute.population.check   # every population/ dataset against its bms/ twin, element for element (W0 identity)

# --- bmstp (the prior: shape grids per grain, the per-source density table, template weights, the atlas; writes bmstp/) ---
# Lines are added here as each stage lands (IMPLEMENTATION_BMSTP sec 3).
PY sesnaimpute.bmstp.shapes   # the star-family, cloud-class and galaxy shape grids, P2/P3/P4 (SPEC_BMSTP sec 4.1, 5.1-5.6)
PY sesnaimpute.bmstp.template_weights   # the P5 template-weight tables per library (SPEC_BMSTP sec 1.4, 4.1, 4.2, 5.1-5.6)
PY sesnaimpute.bmstp.density   # the per-source density table, P1: column, grain indices, F_LIM_50, D_PAHC, the six sky densities (SPEC_BMSTP sec 4.1, 5.1-5.6)
PY sesnaimpute.bmstp.atlas   # the prior atlas, P6: per-pixel Monte Carlo selection and the total-count check (SPEC_BMSTP sec 8)
PY sesnaimpute.atlas.render   # the sky atlas figures from P6 (the prior rows; the posterior row is added by the same line at the fitter block's end once P11 exists)
PY sesnaimpute.atlas.protostars   # the protostar check figure per region with HOPS/eHOPS coverage (report-only, SPEC_BMSTP sec 9)

# --- fittp (the thinned-Poisson fitter, the classification, the cascade, the posterior atlas; writes fittp/) ---
# Lines are added here as each stage lands (IMPLEMENTATION_BMSTP sec 4). The
# [fit]/[fittp] knobs (config.py's [fit] section) live in root.cfg:
# topk -> fit_topk (fittp.sweep's per-source top-K record size), batch_size
# -> fit_batch_size (fittp.sweep's per-part-file source count), block_budget_mb
# -> fit_block_budget_mb (fittp.sweep/likelihood.block_size's per-block memory
# cap), beta -> fit_beta (fittp.classify's cascade-evidence blend weight).
# cascade's measured half needs no fit, so it runs before the fit loop; its
# imputed half (fittp.classify's FLUX_IMPUTED) is filled in by re-running this
# same line after classify, once per region, once classify has written.
PY sesnaimpute.fittp.cascade   # the colour cascade on the measured fluxes, Psi per source (SPEC_BMSTP sec 6.5)
PY sesnaimpute.fittp.library_resolution   # SIGMA_LIB_DEX per library, the fit's per-band variance floor (SPEC_BMSTP sec 6.1)
# One capped.sh process per class, so one class's peak resident is never summed with the class before it.
for FIT_CLASS in STAR AGB PAHC GAL YSO H2S; do
  PY sesnaimpute.fittp.sweep --classes "$FIT_CLASS"   # the class evidence sweep, P7, batched at [fit] batch_size/block_budget_mb, K=[fit] topk (SPEC_BMSTP sec 1.3; IMPLEMENTATION_BMSTP sec 4 row 2.4)
done
PY sesnaimpute.fittp.classify   # P(C|D) at [fit] beta, subclasses, MAP, imputed flux (P8) and the literature-band sensitivity (P9) (SPEC_BMSTP sec 7.1, 7.2)
PY sesnaimpute.fittp.cascade   # re-run: fills the cascade's imputed half now that classify has written (SPEC_BMSTP sec 7.3)
PY sesnaimpute.fittp.atlas   # the posterior atlas, P11: per-pixel mean P(C|D) and the P(YSO)>0.5 count (SPEC_BMSTP sec 8)
PY sesnaimpute.fittp.check   # the spec sec 9 checks read from P1-P11, report only (IMPLEMENTATION_BMSTP sec 7)
PY sesnaimpute.atlas.render   # the sky atlas figures from P6 and, where it exists, P11 -- renders both when both exist (SPEC_BMSTP sec 8)

# A --from that never matched any PY line above would otherwise leave
# every stage silently skipped and the script exiting 0 having done
# nothing -- fail loudly instead.
if [ -n "$FROM" ] && [ $STARTED -eq 0 ]; then
  echo "RUNBOOKtp.sh: --from $FROM matched no stage" >&2
  exit 2
fi
