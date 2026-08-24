#!/bin/bash -l
#SBATCH --job-name=gand_ft
#SBATCH --account=<ACCOUNT>
#SBATCH --cluster=<CLUSTER>
#SBATCH --partition=<PARTITION_A100_80GB>
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"

mkdir -p logs ft_outputs

if [ -z "${CONFIG:-}" ]; then
    echo "ERROR: set CONFIG=ft_configs/ft_<student>_<cond>.yml" >&2
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

# Optional smoke override: pass NUM_EPOCHS / MAX_STEPS / DATASET_LIMIT to the trainer via env.
echo "=================================================="
echo "  GAND fine-tuning job"
echo "=================================================="
echo "  Job ID:    ${SLURM_JOB_ID:-local}"
echo "  Node:      ${SLURMD_NODENAME:-$(hostname)}"
echo "  Project:   $PROJECT_DIR"
echo "  Config:    $CONFIG"
echo "  Overrides: NUM_EPOCHS=${NUM_EPOCHS:-} MAX_STEPS=${MAX_STEPS:-}"
echo "=================================================="

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU inventory:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
fi

module --force purge 2>/dev/null || true
module load cluster/<CLUSTER>/<GPU_MODULE> 2>/dev/null || true
module load CUDA/12.6.0 2>/dev/null || true

# HF cache on project-scratch (<ACCOUNT> is at 100% quota; use 2026_026 which has 909 GB free).
export HF_HOME="${JOB_HF_HOME:-${PROJECT_SCRATCH}/.cache/huggingface}"
mkdir -p "$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$(tr -d '[:space:]' < ${HOME}/.huggingface/token 2>/dev/null)}"
export PYTHONUNBUFFERED=1

CONDA_ENV="${CONDA_ENV:-${PROJECT_SCRATCH}/.venvs/axolotl_latest}"
# The env is venv-style - no conda.sh needed. Just put its bin first on PATH.
export PATH="$CONDA_ENV/bin:$PATH"
# Redirect uv/HF/Triton/CUDA caches off user-scratch (which is at quota).
export UV_CACHE_DIR=${PROJECT_SCRATCH}/.cache/uv
export XDG_CACHE_HOME=${PROJECT_SCRATCH}/.cache
export TRITON_CACHE_DIR=${PROJECT_SCRATCH}/.cache/triton
export CUDA_CACHE_PATH=${PROJECT_SCRATCH}/.cache/nv/ComputeCache
mkdir -p "$UV_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
export AXOLOTL_DO_NOT_TRACK=1
echo "[env] python: $(which python)  ($(python --version 2>&1))"
echo "[env] axolotl: $(which axolotl)"
python -c "import transformers; print(f'[env] transformers: {transformers.__version__}')" 2>&1 | head -1

# Compose extra-axolotl-args (override via env). Most knobs are in the YAML.
EXTRA=()
if [ -n "${NUM_EPOCHS:-}" ]; then EXTRA+=(--num_epochs "$NUM_EPOCHS"); fi
if [ -n "${MAX_STEPS:-}" ]; then EXTRA+=(--max_steps "$MAX_STEPS"); fi

axolotl train "$CONFIG" "${EXTRA[@]}"

echo ""
echo "=================================================="
echo "Done. Adapter: ft_outputs/$(basename "${CONFIG%.yml}")/"
echo "=================================================="
