#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODELS=${MODELS:-"Qwen/Qwen3.6-27B google/gemma-4-31B-it"}
CONDITIONS=${CONDITIONS:-"zero_shot few_shot_full"}
INPUT=${INPUT:-data/test-00000-of-00001_with_variants.parquet}

ACCOUNT=${ACCOUNT:-<ACCOUNT>}
TIME=${TIME:-02:00:00}
DTYPE=${DTYPE:-bfloat16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
LIMIT=${LIMIT:-}
DRY_RUN=${DRY_RUN:-}
SMART=${SMART:-1}
PREFER_TP=${PREFER_TP:-}
TIE_BREAK_WINDOW=${TIE_BREAK_WINDOW:-1800}

PARTITION=${PARTITION:-}
GPUS=${GPUS:-}
TP=${TP:-}

VENV_PYTHON=${VENV_PYTHON:-${PROJECT_SCRATCH}/.conda/envs/data_diversity_generation_venv/bin/python}

mkdir -p results/classify logs

JOB_IDS=()
SUBMITTED=0
SKIPPED=0

pick_slot_for() {
    local model="$1"
    if [ -n "$PARTITION" ] && [ -n "$GPUS" ] && [ -n "$TP" ]; then
        echo "PARTITION=$PARTITION"
        echo "GPUS=$GPUS"
        echo "TP=$TP"
        return 0
    fi
    if [ "$SMART" = "0" ]; then
        echo "PARTITION=${PARTITION:-<PARTITION_A100>}"
        echo "GPUS=${GPUS:-4}"
        echo "TP=${TP:-4}"
        return 0
    fi
    local extra_args=()
    if [ -n "$PREFER_TP" ]; then
        extra_args+=(--prefer-tp "$PREFER_TP")
    fi
    "$VENV_PYTHON" "$PROJECT_DIR/scripts/pick_slot.py" \
        --model "$model" \
        --account "$ACCOUNT" \
        --time "$TIME" \
        --tie-break-window "$TIE_BREAK_WINDOW" \
        "${extra_args[@]}" \
        2> >(sed 's/^/  [pick_slot] /' >&2)
}

for model in $MODELS; do
    safe="$(echo "$model" | tr '/' '_')"
    slot=""
    for cond in $CONDITIONS; do
        out="results/classify/test_${safe}_${cond}.parquet"
        job_name="aaai_${safe}_${cond}"
        if [ -f "$out" ]; then
            echo "[skip] $out already exists"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        if [ -z "$slot" ]; then
            slot="$(pick_slot_for "$model")"
            if [ -z "$slot" ]; then
                echo "[error] pick_slot failed for $model"
                continue
            fi
            eval "$slot"
        fi
        echo "[submit] $model x $cond on $PARTITION (gpus=$GPUS, tp=$TP) -> $out"
        if [ -n "$DRY_RUN" ]; then
            echo "         (DRY_RUN: not submitting)"
            continue
        fi
        export_vars="MODEL=$model,CONDITION=$cond,INPUT=$INPUT,TP=$TP,DTYPE=$DTYPE,\
MAX_MODEL_LEN=$MAX_MODEL_LEN,GPU_MEM_UTIL=$GPU_MEM_UTIL,OUTPUT=$out"
        if [ -n "$LIMIT" ]; then
            export_vars="${export_vars},LIMIT=$LIMIT"
        fi
        jobid_raw=$(sbatch --parsable \
            --cluster=<CLUSTER> \
            --partition="$PARTITION" \
            --account="$ACCOUNT" \
            --gres="gpu:$GPUS" \
            --time="$TIME" \
            --job-name="$job_name" \
            --export="ALL,$export_vars" \
            scripts/run_local_classify.slurm)
        jobid="${jobid_raw%%;*}"
        JOB_IDS+=("$jobid")
        SUBMITTED=$((SUBMITTED + 1))
        echo "         submitted job $jobid"
    done
done

echo ""
echo "=================================================="
echo "  Sweep summary"
echo "=================================================="
echo "  Submitted: $SUBMITTED"
echo "  Skipped:   $SKIPPED"
if [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo "  Job IDs:   ${JOB_IDS[*]}"
    echo ""
    csv_ids="$(IFS=,; echo "${JOB_IDS[*]}")"
    echo "Monitor with:"
    echo "  squeue --cluster=<CLUSTER> -u \$USER"
    echo "  squeue --cluster=<CLUSTER> -j $csv_ids"
    echo "  sacct --cluster=<CLUSTER> -j $csv_ids --format=JobID,JobName,State,Elapsed"
    echo "  tail -f logs/*_${JOB_IDS[0]}.out"
fi
