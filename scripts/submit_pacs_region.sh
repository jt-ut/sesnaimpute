#!/bin/bash
# Submits one region's staged PACS completeness (sesnaimpute.sky.derived.pacs_stages) as 2 + 3 jobs per band, each heavy
# job spread over as many nodes as its work needs, so that every job takes about the same time.
#
#   submit_pacs_region.sh [--phase 1|2] [--dry-run [--tasks P70=N,P100=N,P160=N]] [--box RA DEC W H] [--reuse-mosaics]
#                         [--reference-product P] [--target-minutes M] SYSTEM REGION CONFIG
#
#   Two phases, both run on a LOGIN node (TACC refuses sbatch from a compute node: "Job submission is not allowed from
#   this host", smoke job 3661756):
#   --phase 1 (default)  prepare and the three detect jobs
#   --phase 2            after the detect jobs have ended: each band's K from its detect.pkl (`pacs_stages plan`, read
#                        here), then its inject job (K nodes); prints "INJECT <band> K=<K> ID=<job id>"
#   --phase 3 --band B --k K   the band's pool job (K nodes), once its inject job has ended
#   --phase 4            finish, once every band's pool job has ended
#   (a job waits in the queue only when it can run: a 7-job chain per region held 7 of the 36 submissions per user)
#
#   SYSTEM  vista (gg, 144 workers per node) or stampede3 (spr, 112 workers per node)
#   --target-minutes  wanted minutes of an injection or pooling job (default 45); sets the nodes K of each band
#   --dry-run  prints every sbatch line and submits nothing; phase 2 then takes the bands' round-0 task counts from
#              --tasks instead of their detect.pkl
#
# Jobs (`pacs_sizing` has the node counts and the -t of every job: 3 x its expected minutes, rounded up to 30 min):
#   prepare                         1 node: preflight + mosaics
#     detect <band>                 1 node (afterok prepare)                                   -- phase 1
#       inject <band>               K nodes: rounds 0 .. MAX_EXTRA_BATCHES, the K shares of a round together (one node
#                                   each, by srun), a barrier between rounds                   -- phase 2
#         pool <band>               K nodes (afterok inject): pool tables, the K pool shares together, assemble
#           finish                  1 node (afterok every pool job): product, sources, acceptances, photometry
# A job whose output file exists does nothing, so a failed job is resubmitted alone or its phase whole.
# The job body is this same file, run by sbatch as `submit_pacs_region.sh --job KIND ...`.

ROOT=/work2/08772/jjt277/stampede3
SLURM_DIR=$ROOT/projects/SESNA_Complete/pacs_slurm
SOFTWARE=$ROOT/software/sesnaimpute
BANDS="P70 P100 P160"
ABS=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")

DRY=0; BOX=(); REUSE=(); REFERENCE=(); TARGET=45; TASKS=""; JOB=""; BAND=""; K=1; PHASE=1; POOL_T=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --box) BOX=(--box "$2" "$3" "$4" "$5"); shift 5 ;;
        --reuse-mosaics) REUSE=(--reuse-mosaics); shift ;;
        --reference-product) REFERENCE=(--reference-product "$2"); shift 2 ;;
        --target-minutes) TARGET="$2"; shift 2 ;;
        --job) JOB="$2"; shift 2 ;;
        --phase) PHASE="$2"; shift 2 ;;
        --pool-t) POOL_T="$2"; shift 2 ;;
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
        detect)  stage detect --band $BAND || exit 1 ;;
        inject)  stage inject-band --band $BAND --of $K --launcher srun --cpus $CPUS || exit 1 ;;
        pool)    stage pool-band --band $BAND --of $K --launcher srun --cpus $CPUS || exit 1 ;;
        *) echo "unknown job $JOB" >&2; exit 2 ;;
    esac
    exit 0
fi

# ---- submission (a login node)
eval "$(sizing --times | sed -n 's/^TIMES //p')"
QUEUED=0
[ $DRY -eq 1 ] || QUEUED=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
if [ "$PHASE" = 1 ]; then
    [ $((4 + QUEUED)) -le $MAX_SUBMIT ] || { echo "refused: 4 jobs (+ $QUEUED already queued) against $MAX_SUBMIT submitted on $SYSTEM" >&2; exit 1; }
    echo "# $SYSTEM $PART phase 1: prepare, 3 x detect; run --phase 2 once the detect jobs have ended"
    PREP=$(submit "pacs_${TAG}_prepare" 1 $PREPARE_T "" prepare)
    for b in $BANDS; do
        eval "T=\$DETECT_${b}_T"
        submit "pacs_${TAG}_${b}_detect" 1 $T "$PREP" detect --band $b
    done
elif [ "$PHASE" = 2 ]; then
    [ $((3 + QUEUED)) -le $MAX_SUBMIT ] || { echo "refused: 3 jobs (+ $QUEUED already queued) against $MAX_SUBMIT submitted on $SYSTEM" >&2; exit 1; }
    if [ $DRY -eq 1 ]; then
        [ -n "$TASKS" ] || { echo "--dry-run --phase 2 needs --tasks P70=N,P100=N,P160=N" >&2; exit 2; }
        SIZES=$(sizing --target-minutes $TARGET --tasks "$TASKS")
    else
        # the plan from the bands' detect.pkl, read on this login node (light: the detect files' task lists)
        source /home1/08772/jjt277/startups/sesnaimpute_source.sh
        SIZES=""
        for b in $BANDS; do
            SIZES+=$(PYTHONPATH=$SOFTWARE/src python -W ignore -m sesnaimpute.sky.derived.pacs_stages plan --config "$CONFIG" --region "$REGION" \
                     --band $b "${BOX[@]}" --system $SYSTEM --target-minutes $TARGET) || { echo "plan of $b failed (has its detect job ended?)" >&2; exit 1; }
            SIZES+=$'\n'
        done
    fi
    echo "$SIZES" | grep -i "^plan" >&2
    echo "# $SYSTEM $PART phase 2: 3 x inject (pool: --phase 3 per band as its inject ends; finish: --phase 4)"
    for b in $BANDS; do
        L=$(echo "$SIZES" | grep "^PLAN $b ")
        KB=$(field K "$L"); IT=$(field INJECT_T "$L"); PT=$(field POOL_T "$L")
        [ -n "$KB" ] && [ -n "$IT" ] && [ -n "$PT" ] || { echo "no plan line for $b" >&2; exit 1; }
        INJ=$(submit "pacs_${TAG}_${b}_inject" $KB $IT "" inject --band $b --k $KB)
        echo "INJECT $b K=$KB POOL_T=$PT ID=$INJ"
    done
elif [ "$PHASE" = 3 ]; then
    [ -n "$BAND" ] && [ "$K" -ge 1 ] && [ -n "$POOL_T" ] || { echo "--phase 3 needs --band, --k and --pool-t (phase 2's POOL_T)" >&2; exit 2; }
    [ $((1 + QUEUED)) -le $MAX_SUBMIT ] || { echo "refused: 1 job (+ $QUEUED already queued) against $MAX_SUBMIT" >&2; exit 1; }
    echo "POOL $BAND K=$K ID=$(submit "pacs_${TAG}_${BAND}_pool" $K $POOL_T "" pool --band $BAND --k $K)"
elif [ "$PHASE" = 4 ]; then
    [ $((1 + QUEUED)) -le $MAX_SUBMIT ] || { echo "refused: 1 job (+ $QUEUED already queued) against $MAX_SUBMIT" >&2; exit 1; }
    echo "FINISH ID=$(submit "pacs_${TAG}_finish" 1 $FINISH_T "" finish)"
else
    echo "--phase must be 1, 2, 3 or 4" >&2; exit 2
fi
