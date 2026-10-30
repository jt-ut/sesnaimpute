#!/bin/bash
# Submits one region's staged PACS completeness (sesnaimpute.sky.derived.pacs_stages) as 2 + 3 jobs per band, each heavy
# job spread over as many nodes as its work needs, so that every job takes about the same time.
#
#   submit_pacs_region.sh [--dry-run --tasks P70=N,P100=N,P160=N] [--box RA DEC W H] [--reuse-mosaics]
#                         [--reference-product P] [--target-minutes M] SYSTEM REGION CONFIG
#
#   SYSTEM  vista (gg, 144 workers per node) or stampede3 (spr, 112 workers per node)
#   --target-minutes  wanted minutes of an injection or pooling job (default 45); sets the nodes K of each band
#   --dry-run  prints every sbatch line and submits nothing; the bands' round-0 task counts (which the detect jobs read
#              from detect.pkl) are given by --tasks, and the jobs the detect jobs would submit are printed too
#
# Jobs (`pacs_sizing` has the node counts and the -t of every job: 3 x its expected minutes, rounded up to 30 min):
#   prepare                         1 node: preflight + mosaics
#     detect <band>                 1 node; at its end plans the band's K from detect.pkl and submits the next two itself
#       inject <band>               K nodes: rounds 0 .. MAX_EXTRA_BATCHES, the K shares of a round together (one node
#                                   each, by srun), a barrier between rounds
#         pool <band>               K nodes (afterok inject): pool tables, the K pool shares together, assemble; the
#                                   pool job that completes the last band submits finish (claimed by a marker file,
#                                   so exactly one does)
#           finish                  1 node: product, sources, acceptances, photometry
# A job whose output file exists does nothing, so a failed chain is resubmitted whole (a failed finish is submitted
# alone: `submit_pacs_region.sh --job finish ...` needs sbatch; or resubmit the chain after removing the marker
# <stage store>/finish_submitted).
# The job body is this same file, run by sbatch as `submit_pacs_region.sh --job KIND ...`.

ROOT=/work2/08772/jjt277/stampede3
SLURM_DIR=$ROOT/projects/SESNA_Complete/pacs_slurm
SOFTWARE=$ROOT/software/sesnaimpute-pacs_injection
BANDS="P70 P100 P160"
ABS=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")

DRY=0; BOX=(); REUSE=(); REFERENCE=(); TARGET=45; TASKS=""; JOB=""; BAND=""; K=1
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --box) BOX=(--box "$2" "$3" "$4" "$5"); shift 5 ;;
        --reuse-mosaics) REUSE=(--reuse-mosaics); shift ;;
        --reference-product) REFERENCE=(--reference-product "$2"); shift 2 ;;
        --target-minutes) TARGET="$2"; shift 2 ;;
        --job) JOB="$2"; shift 2 ;;
        --band) BAND="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
[ $# -eq 3 ] || { sed -n '2,9p' "$0" >&2; exit 2; }
SYSTEM=$1; REGION=$2; CONFIG=$3

case "$SYSTEM" in
    vista)     PART=gg;  CPUS=144; MAX_SUBMIT=40; EXTRA=(--propagate=NONE) ;;
    stampede3) PART=spr; CPUS=112; MAX_SUBMIT=36; EXTRA=() ;;
    *) echo "system must be vista or stampede3" >&2; exit 2 ;;
esac

# the options every job of the chain is given again (the job body parses the same line)
COMMON=(--target-minutes "$TARGET" "${BOX[@]}" "${REUSE[@]}" "${REFERENCE[@]}")
# the pacs_stages command line of a stage, in a job
stage() { python -W ignore -m sesnaimpute.sky.derived.pacs_stages "$1" --config "$CONFIG" --region "$REGION" --n-jobs $CPUS "${BOX[@]}" "${@:2}"; }
TAG=$(echo "$REGION" | tr -c 'A-Za-z0-9\n' _)

# submit NAME NODES TIME DEPENDENCY(afterok list or empty) job-args...  -> prints the job id
submit() {
    local name=$1 nodes=$2 time=$3 dep=$4; shift 4
    local cmd=(sbatch --parsable -J "$name" -o "$SLURM_DIR/%x_%j.out" -e "$SLURM_DIR/%x_%j.err" -p $PART -N $nodes -n $nodes -c $CPUS
               -t $time -A AST24021 "${EXTRA[@]}")
    [ -z "$dep" ] || cmd+=(--dependency="afterok:$dep")
    cmd+=("$ABS" "${COMMON[@]}" --job "$@" "$SYSTEM" "$REGION" "$CONFIG")
    if [ $DRY -eq 1 ]; then
        { printf '%q ' "${cmd[@]}"; echo; } >&2     # the dry-run job id is the job's name
        echo "$name"
    else
        # sbatch on Stampede3 prints its banner on stdout too: the id is the last all-digit line
        local id
        id=$("${cmd[@]}" | cut -d';' -f1 | grep -E '^[0-9]+$' | tail -1)
        [ -n "$id" ] || { echo "submission of $name failed: stopping (cancel this chain's earlier jobs: squeue -u $USER)" >&2; kill -TERM $$; exit 1; }
        echo "$id"
    fi
}
# sizing ARGS...: pacs_sizing of this checkout (no numbers or cluster software needed)
sizing() { PYTHONPATH="$(dirname "$ABS")/../src" ${DRY_PYTHON:-python3} -m sesnaimpute.sky.derived.pacs_sizing --system $SYSTEM "$@"; }
# field NAME LINE: the value of NAME=value in a plan line
field() { echo "$2" | tr ' ' '\n' | sed -n "s/^$1=//p"; }

if [ -n "$JOB" ]; then
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
    # glibc: a fixed mmap threshold (no dynamic raise) so the workers' large temporaries go back to the system when
    # freed (job 3661209: worker memory grew per piece; not reproduced on macOS)
    export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
    ulimit -u $(ulimit -Hu)
    source /home1/08772/jjt277/startups/sesnaimpute_source.sh
    export PYTHONPATH=$SOFTWARE/src
    cd $SLURM_DIR
    python -c "import numpy, sys; sys.exit(0 if numpy.__version__ == \"1.24.4\" else 1)" || { echo "numpy is not 1.24.4"; exit 1; }
    python -W ignore -c "import sesnaimpute; print(\"code:\", sesnaimpute.__file__)"
    ( cd $SOFTWARE && git log --oneline -1 )
    echo "job $JOB $BAND on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST"
    case "$JOB" in
        prepare) exec python -W ignore -m sesnaimpute.sky.derived.pacs_stages preflight --config "$CONFIG" --region "$REGION" --n-jobs $CPUS "${BOX[@]}" "${REUSE[@]}" "${REFERENCE[@]}" ;;
        finish)  exec python -W ignore -m sesnaimpute.sky.derived.pacs_stages finish --config "$CONFIG" --region "$REGION" --n-jobs $CPUS "${BOX[@]}" "${REFERENCE[@]}" ;;
        detect)
            stage detect --band $BAND || exit 1
            PLAN=$(stage plan --band $BAND --system $SYSTEM --target-minutes $TARGET) || exit 1
            echo "$PLAN"
            LINE=$(echo "$PLAN" | grep "^PLAN $BAND ")
            KB=$(field K "$LINE"); IT=$(field INJECT_T "$LINE"); PT=$(field POOL_T "$LINE")
            [ -n "$KB" ] && [ -n "$IT" ] && [ -n "$PT" ] || { echo "no plan line for $BAND"; exit 1; }
            INJ=$(submit "pacs_${TAG}_${BAND}_inject" $KB $IT "" inject --band $BAND --k $KB) || exit 1
            submit "pacs_${TAG}_${BAND}_pool" $KB $PT "$INJ" pool --band $BAND --k $KB >/dev/null || exit 1 ;;
        inject)  stage inject-band --band $BAND --of $K --launcher srun --cpus $CPUS || exit 1 ;;
        pool)
            stage pool-band --band $BAND --of $K --launcher srun --cpus $CPUS || exit 1
            stage claim-finish; RC=$?
            if [ $RC -eq 0 ]; then
                eval "$(sizing --times | sed -n 's/^TIMES //p')"
                submit "pacs_${TAG}_finish" 1 $FINISH_T "" finish >/dev/null || exit 1
            elif [ $RC -ne 3 ]; then
                exit $RC
            fi ;;
        *) echo "unknown job $JOB" >&2; exit 2 ;;
    esac
    exit 0
fi

# ---- submission
eval "$(sizing --times | sed -n 's/^TIMES //p')"
if [ $DRY -eq 1 ]; then
    [ -n "$TASKS" ] || { echo "--dry-run needs --tasks P70=N,P100=N,P160=N (the round-0 task counts the detect jobs read)" >&2; exit 2; }
    SIZES=$(sizing --target-minutes $TARGET --tasks "$TASKS")
else
    SIZES=""
    QUEUED=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
    [ $((11 + QUEUED)) -le $MAX_SUBMIT ] || { echo "refused: 11 jobs (+ $QUEUED already queued) against $MAX_SUBMIT submitted on $SYSTEM" >&2; exit 1; }
fi
echo "# $SYSTEM $PART: 11 jobs (prepare, 3 x detect, 3 x inject, 3 x pool, finish); inject and pool are submitted by the band's detect job"

PREP=$(submit "pacs_${TAG}_prepare" 1 $PREPARE_T "" prepare)
for b in $BANDS; do
    eval "T=\$DETECT_${b}_T"
    DET=$(submit "pacs_${TAG}_${b}_detect" 1 $T "$PREP" detect --band $b)
    if [ $DRY -eq 1 ]; then
        echo "$SIZES" | grep "^plan $b" >&2
        L=$(echo "$SIZES" | grep "^PLAN $b ")
        KB=$(field K "$L")
        echo "# submitted by $DET at its end:" >&2
        INJ=$(submit "pacs_${TAG}_${b}_inject" $KB $(field INJECT_T "$L") "" inject --band $b --k $KB)
        submit "pacs_${TAG}_${b}_pool" $KB $(field POOL_T "$L") "$INJ" pool --band $b --k $KB >/dev/null
    fi
done
if [ $DRY -eq 1 ]; then
    echo "# submitted by the pool job that completes the last band:" >&2
    submit "pacs_${TAG}_finish" 1 $FINISH_T "" finish >/dev/null
fi
