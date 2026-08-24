#!/usr/bin/env bash
# Run a single condition across multiple open-source models.
# Override MODELS / CONDITION / BACKEND via env.
set -euo pipefail

MODELS=${MODELS:-"meta-llama/Meta-Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen2.5-7B-Instruct"}
CONDITION=${CONDITION:-zero_shot few_shot few_shot_full}
INPUT=${INPUT:-data/test-00000-of-00001_with_variants.parquet}
BACKEND=${BACKEND:-transformers}

mkdir -p results/classify

for model in $MODELS; do
  for cond in $CONDITION; do
    safe_name=$(echo "$model" | tr '/' '_')
    out="results/classify/test_${safe_name}_${cond}.parquet"
    if [ -f "$out" ]; then
      echo "[skip] $out already exists"
      continue
    fi
    echo "[run] $model / $cond -> $out"
    python classify_gender_local.py \
        --input "$INPUT" \
        --model "$model" \
        --condition "$cond" \
        --output "$out" \
        --backend "$BACKEND"
  done
done
