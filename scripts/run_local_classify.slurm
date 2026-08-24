#!/bin/bash -l
#SBATCH --job-name=gand_classify
#SBATCH --account=<ACCOUNT>
#SBATCH --cluster=<CLUSTER>
#SBATCH --partition=<PARTITION_A100>
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"

mkdir -p logs results/classify .cache/huggingface .cache/vllm

MODEL="${MODEL:?ERROR: set MODEL=<HF model id>}"
CONDITION="${CONDITION:?ERROR: set CONDITION=<zero_shot|few_shot|few_shot_full|...>}"
INPUT="${INPUT:-data/test-00000-of-00001_with_variants.parquet}"
TP="${TP:-4}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
MAX_PARALLEL="${MAX_PARALLEL:-64}"
LIMIT="${LIMIT:-}"
CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-}"

if [[ "$MODEL" == Qwen/Qwen3* ]] && [ -z "$CHAT_TEMPLATE_KWARGS" ]; then
    CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
fi

if [ -z "$MAX_NEW_TOKENS" ]; then
    if [[ "$CHAT_TEMPLATE_KWARGS" == *"\"enable_thinking\": true"* ]]; then
        MAX_NEW_TOKENS=4096
    else
        MAX_NEW_TOKENS=1024
    fi
fi

MODEL_SAFE="$(echo "$MODEL" | tr '/' '_')"
OUTPUT="${OUTPUT:-results/classify/test_${MODEL_SAFE}_${CONDITION}.parquet}"

SIF="${SIF:-$PROJECT_DIR/.cache/singularity/vllm-openai-latest.sif}"
# bashrc may set HF_HOME to a stale path; prefer JOB_HF_HOME passed via --export.
unset HF_HOME
HF_HOME="${JOB_HF_HOME:-${PROJECT_SCRATCH}/.cache/huggingface}"
if [ ! -r "$HF_HOME" ] || [ ! -w "$HF_HOME" ]; then
    echo "[WARN] HF_HOME=$HF_HOME is not readable+writable; falling back to \$PROJECT_DIR/.cache/huggingface"
    HF_HOME="$PROJECT_DIR/.cache/huggingface"
fi
mkdir -p "$HF_HOME"
export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1

VENV_PYTHON=${PROJECT_SCRATCH}/.conda/envs/data_diversity_generation_venv/bin/python

echo "=================================================="
echo "  GAND classification job"
echo "=================================================="
echo "  Job ID:         ${SLURM_JOB_ID:-local}"
echo "  Node:           ${SLURMD_NODENAME:-$(hostname)}"
echo "  Project dir:    $PROJECT_DIR"
echo "  Model:          $MODEL"
echo "  Condition:      $CONDITION"
echo "  Input:          $INPUT"
echo "  Output:         $OUTPUT"
echo "  TP:             $TP"
echo "  dtype:          $DTYPE"
echo "  max_model_len:  $MAX_MODEL_LEN"
echo "  gpu_mem_util:   $GPU_MEM_UTIL"
echo "  max_new_tokens: $MAX_NEW_TOKENS"
echo "  max_parallel:   $MAX_PARALLEL"
echo "  limit:          ${LIMIT:-(none)}"
echo "  chat_template_kwargs: ${CHAT_TEMPLATE_KWARGS:-(none)}"
echo "  SIF:            $SIF"
echo "  HF_HOME:        $HF_HOME"
echo "=================================================="

if [ ! -f "$SIF" ]; then
    echo "ERROR: Singularity image not found at $SIF" >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU inventory:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
    echo "[WARN] nvidia-smi not on PATH"
fi

module --force purge 2>/dev/null || true
module load "cluster/<CLUSTER>/${SLURM_JOB_PARTITION:-<PARTITION_A100>}" 2>/dev/null || true
module load CUDA/12.6.0 2>/dev/null || true

PORT="$(( 20000 + RANDOM % 20000 ))"
HOST_IP="$(hostname -I | awk '{print $1}')"
BASE_URL="http://localhost:${PORT}"

echo ""
echo "Starting vLLM server on port $PORT (TP=$TP)..."

unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE \
      LD_LIBRARY_PATH PYTHONPATH PYTHONHOME

CACHE_ROOT="$PROJECT_DIR/.cache/container_root"
mkdir -p "$CACHE_ROOT/.cache/vllm" "$CACHE_ROOT/.cache/triton" \
         "$CACHE_ROOT/.cache/huggingface" "$CACHE_ROOT/.nv" \
         "$CACHE_ROOT/tmp"

VLLM_LORA_ARGS=()
if [ -n "${LORA_NAME:-}" ] && [ -n "${LORA_PATH:-}" ]; then
    VLLM_LORA_ARGS+=(
        --enable-lora
        --lora-modules "${LORA_NAME}=${LORA_PATH}"
        --max-lora-rank "${MAX_LORA_RANK:-64}"
        --max-loras 1
    )
    echo "[lora] vLLM will serve adapter ${LORA_PATH} as alias '${LORA_NAME}' (rank ${MAX_LORA_RANK:-64})"
fi

singularity exec --nv --cleanenv \
    --env HOME=/root \
    --env XDG_CACHE_HOME=/root/.cache \
    --env HF_HUB_CACHE="/root/.cache/huggingface/hub" \
    --env HF_HUB_ENABLE_HF_TRANSFER=0 \
    --env VLLM_CACHE_ROOT=/root/.cache/vllm \
    --env TRITON_CACHE_DIR=/root/.cache/triton \
    --env CUDA_CACHE_PATH=/root/.nv/ComputeCache \
    --env TMPDIR=/tmp \
    --env PYTHONUNBUFFERED=1 \
    --env "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
    --bind "$CACHE_ROOT:/root" \
    --bind "$HF_HOME:/root/.cache/huggingface" \
    --bind "$PROJECT_DIR/.cache/vllm:/root/.cache/vllm" \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    "$SIF" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --tensor-parallel-size "$TP" \
        --dtype "$DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --trust-remote-code \
        "${VLLM_LORA_ARGS[@]}" \
    > "logs/vllm_${SLURM_JOB_ID:-local}.log" 2>&1 &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID  log: logs/vllm_${SLURM_JOB_ID:-local}.log"

cleanup() {
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Cleaning up vLLM server (PID $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Waiting for /health on $BASE_URL ..."
MAX_WAIT=1800
ELAPSED=0
while true; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM process exited prematurely. Tail of log:" >&2
        tail -60 "logs/vllm_${SLURM_JOB_ID:-local}.log" >&2
        exit 1
    fi
    if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
        echo "vLLM is healthy after ${ELAPSED}s"
        break
    fi
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: vLLM did not become healthy within ${MAX_WAIT}s" >&2
        tail -60 "logs/vllm_${SLURM_JOB_ID:-local}.log" >&2
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ $((ELAPSED % 60)) -eq 0 ]; then
        echo "  ...still waiting (${ELAPSED}s)"
    fi
done

echo ""
echo "Running classification client..."
LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

CTK_ARG=()
if [ -n "$CHAT_TEMPLATE_KWARGS" ]; then
    CTK_ARG=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")
fi

SAMPLING_ARG=()
if [ -n "${TEMPERATURE:-}" ]; then
    SAMPLING_ARG+=(--temperature "$TEMPERATURE")
fi
if [ -n "${TOP_P:-}" ]; then
    SAMPLING_ARG+=(--top-p "$TOP_P")
fi
if [ -n "${SYSTEM_PROMPT_FILE:-}" ]; then
    SAMPLING_ARG+=(--system-prompt-file "$SYSTEM_PROMPT_FILE")
fi

FEWSHOT_ARG=()
if [ -n "${FEWSHOT_TRAIN:-}" ]; then
    FEWSHOT_ARG+=(--fewshot-train-input "$FEWSHOT_TRAIN")
fi
if [ -n "${FEWSHOT_PER_CLASS:-}" ]; then
    FEWSHOT_ARG+=(--fewshot-per-class "$FEWSHOT_PER_CLASS")
fi
if [ -n "${FEWSHOT_SEED:-}" ]; then
    FEWSHOT_ARG+=(--fewshot-seed "$FEWSHOT_SEED")
fi
if [ -n "${FEWSHOT_INCLUDE_THINK:-}" ]; then
    FEWSHOT_ARG+=(--fewshot-include-think)
fi
if [ -n "${FEWSHOT_AMBIGUOUS_ONLY:-}" ]; then
    FEWSHOT_ARG+=(--fewshot-ambiguous-only)
fi

LORA_ARG=()
if [ -n "${LORA_NAME:-}" ]; then
    LORA_ARG+=(--lora-name "$LORA_NAME")
fi

"$VENV_PYTHON" classify_gender_local.py \
    --input "$INPUT" \
    --model "$MODEL" \
    --condition "$CONDITION" \
    --output "$OUTPUT" \
    --backend openai-api \
    --base-url "$BASE_URL" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-parallel "$MAX_PARALLEL" \
    "${CTK_ARG[@]}" \
    "${SAMPLING_ARG[@]}" \
    "${FEWSHOT_ARG[@]}" \
    "${LORA_ARG[@]}" \
    $LIMIT_ARG

echo ""
echo "=================================================="
echo "Done. Output: $OUTPUT"
echo "=================================================="
