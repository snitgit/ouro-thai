#!/bin/bash
#SBATCH --job-name=ouro-thai
#SBATCH --partition=defq
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=40:00:00
#SBATCH --output=slurm/logs/%j_%x.out
#SBATCH --error=slurm/logs/%j_%x.err

# ── Configuration (override via environment) ──
CONFIG=${CONFIG:-configs/a100.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-./output/ouro-2.6b-thai-ut4}
REPORT_TO=${REPORT_TO:-wandb}
TOTAL_UT_STEPS=${TOTAL_UT_STEPS:-}
SIF_IMAGE=${SIF_IMAGE:-ouro.sif}
unset https_proxy

# ── Paths ──
#PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
PROJECT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
HF_CACHE=${HF_HOME:-$HOME/.cache/huggingface}
SCRATCH_DIR=${SCRATCH:-/scratch/$USER}

# Create output dirs
mkdir -p "$OUTPUT_DIR" "$HF_CACHE" slurm/logs

echo "=== Job $SLURM_JOB_ID ==="
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "Current Workspace: $PROJECT_DIR"

# ── Run training ──
module load singularity
singularity exec --nv --contain --writable-tmpfs \
    --bind "$PROJECT_DIR:/workspace" \
    --bind "$HF_CACHE:/scratch/huggingface" \
    --bind "$OUTPUT_DIR:/workspace/output" \
    "$SIF_IMAGE" \
    bash -c "cd /workspace && python train_qlora.py \
        --config $CONFIG \
        --output_dir /workspace/output \
        --report_to $REPORT_TO \
        ${TOTAL_UT_STEPS:+--total_ut_steps $TOTAL_UT_STEPS}"

echo "=== Job $SLURM_JOB_ID finished ==="
