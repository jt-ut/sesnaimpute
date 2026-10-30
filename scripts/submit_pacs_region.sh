#!/bin/bash
# Submits one region's staged PACS completeness (sesnaimpute.sky.derived.pacs_stages) as a chain of jobs.
#
#   submit_pacs_region.sh [--dry-run] [--box RA DEC W H] [--reuse-mosaics] [--reference-product P]
#                         [--k P70=4,P100=1,P160=4] SYSTEM REGION CONFIG
#
#   SYSTEM  vista (gg, 144 workers) or stampede3 (spr, 112 workers)
#   --k     shards K per band (default 1 each); choose with `pacs_stages plan` after the DETECT files exist
#   --dry-run  prints every sbatch line, with its --dependency, and submits nothing
#
# Chain (every job -t 48:00:00, only the time used is charged; one node each):
#   prepare (preflight + mosaics)
#     -> detect <band>                                   one job per band
#       -> inject round 0, shards 0..K-1                 K jobs per band
#         -> inject-rest <band>                          rounds 1..40, one job: each is one small batch behind a barrier
#           -> pool shards 0..K-1                        K jobs per band
#             -> assemble <band>
#               -> finish                                after all bands
# Each dependency is afterok. A job whose output file exists does nothing, so a failed chain is resubmitted whole.
# Queue limits per user (Stampede3 spr 24 running / 36 submitted; Vista gg 20 running / 40 submitted): a chain is
# 2 + sum over bands of (2K + 3) jobs, at most sum(K) of them running together; the script refuses a chain above
# the limits (counting the user's jobs already queued) and names the K to lower.
# The job body is this same file, run by sbatch as `submit_pacs_region.sh --job ...`.

ROOT=/work2/08772/jjt277/stampede3
SLURM_DIR=$ROOT/projects/SESNA_Complete/pacs_slurm
SOFTWARE=$ROOT/software/sesnaimpute-pacs_injection
BANDS="P70 P100 P160"

if [ "${1:-}" = "--job" ]; then
    shift
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
    ulimit -u $(ulimit -Hu)
    source /home1/08772/jjt277/startups/sesnaimpute_source.sh
    export PYTHONPATH=$SOFTWARE/src
    cd $SLURM_DIR
    python -c "import numpy, sys; sys.exit(0 if numpy.__version__ == \"1.24.4\" else 1)" || { echo "numpy is not 1.24.4"; exit 1; }
    python -W ignore -c "import sesnaimpute; print(\"code:\", sesnaimpute.__file__)"
    ( cd $SOFTWARE && git log --oneline -1 )
    exec python -W ignore -m sesnaimpute.sky.derived.pacs_stages "$@"
fi

DRY=0; BOX=(); REUSE=(); REFERENCE=(); KSPEC=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --box) BOX=(--box "$2" "$3" "$4" "$5"); shift 5 ;;
        --reuse-mosaics) REUSE=(--reuse-mosaics); shift ;;
        --reference-product) REFERENCE=(--reference-product "$2"); shift 2 ;;
        --k) KSPEC="$2"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
[ $# -eq 3 ] || { sed -n '2,8p' "$0" >&2; exit 2; }
SYSTEM=$1; REGION=$2; CONFIG=$3

case "$SYSTEM" in
    vista)     PART=gg;  CPUS=144; MAX_SUBMIT=40; MAX_RUN=20; EXTRA=(--propagate=NONE) ;;
    stampede3) PART=spr; CPUS=112; MAX_SUBMIT=36; MAX_RUN=24; EXTRA=() ;;
    *) echo "system must be vista or stampede3" >&2; exit 2 ;;
esac

for b in $BANDS; do eval "K_$b=1"; done
for kv in ${KSPEC//,/ }; do
    b=${kv%%=*}; n=${kv#*=}
    [[ " $BANDS " == *" $b "* && "$n" =~ ^[1-9][0-9]*$ ]] || { echo "bad --k entry $kv (BAND=K, K >= 1)" >&2; exit 2; }
    eval "K_$b=$n"
done

kof() { eval "echo \$K_$1"; }
TOTAL=2; PEAK=0
for b in $BANDS; do k=$(kof $b); TOTAL=$((TOTAL + 2 * k + 3)); PEAK=$((PEAK + k)); done
QUEUED=0
[ $DRY -eq 1 ] || QUEUED=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
if [ $((TOTAL + QUEUED)) -gt $MAX_SUBMIT ] || [ $PEAK -gt $MAX_RUN ]; then
    echo "refused: $TOTAL jobs (+ $QUEUED already queued) against $MAX_SUBMIT submitted, $PEAK running together against $MAX_RUN on $SYSTEM; lower K" >&2
    exit 1
fi
echo "# $SYSTEM $PART: $TOTAL jobs, at most $PEAK running together, $QUEUED already queued"

TAG=$(echo "$REGION" | tr -c 'A-Za-z0-9\n' _)
# submit NAME DEPENDENCY(afterok list or empty) args...  -> prints the job id
submit() {
    local name=$1 dep=$2; shift 2
    local cmd=(sbatch --parsable -J "$name" -o "$SLURM_DIR/%x_%j.out" -e "$SLURM_DIR/%x_%j.err" -p $PART -N 1 -n 1 -c $CPUS
               -t 48:00:00 -A AST24021 "${EXTRA[@]}")
    [ -z "$dep" ] || cmd+=(--dependency="afterok:$dep")
    cmd+=("$ABS" --job "$@" --config "$CONFIG" --region "$REGION" --n-jobs $CPUS "${BOX[@]}")
    if [ $DRY -eq 1 ]; then
        { printf '%q ' "${cmd[@]}"; echo; } >&2     # the dry-run job id is the job's name
        echo "$name"
    else
        "${cmd[@]}" | cut -d';' -f1
    fi
}
ABS=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")

join() { local IFS=:; echo "$*"; }

PREP=$(submit "pacs_${TAG}_prepare" "" preflight "${REUSE[@]}" "${REFERENCE[@]}")
ASM=()
for b in $BANDS; do
    k=$(kof $b)
    DET=$(submit "pacs_${TAG}_${b}_detect" "$PREP" detect --band $b)
    I0=()
    for ((s = 0; s < k; s++)); do
        I0+=($(submit "pacs_${TAG}_${b}_i0_s${s}of${k}" "$DET" inject --band $b --round 0 --shard $s --of $k))
    done
    REST=$(submit "pacs_${TAG}_${b}_irest" "$(join "${I0[@]}")" inject-rest --band $b --round 1 --of $k)
    PL=()
    for ((s = 0; s < k; s++)); do
        PL+=($(submit "pacs_${TAG}_${b}_pool_s${s}of${k}" "$REST" pool --band $b --shard $s --of $k))
    done
    ASM+=($(submit "pacs_${TAG}_${b}_assemble" "$(join "${PL[@]}")" assemble --band $b))
done
submit "pacs_${TAG}_finish" "$(join "${ASM[@]}")" finish "${REFERENCE[@]}" >/dev/null
