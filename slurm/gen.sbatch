#!/bin/bash
#SBATCH --job-name=mini-eg
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/logs/%j_%x.out
#SBATCH --error=slurm/logs/%j_%x.err

# ── Configuration (override via environment) ──
ADAPTER_PATH=${ADAPTER_PATH:-}          # empty = base model
TOTAL_UT_STEPS=${TOTAL_UT_STEPS:-4}
PROMPTS_FILE=${PROMPTS_FILE:-prompts/thai_instruct_bench.json}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
OUTPUT_FILE=${OUTPUT_FILE:-/workspace/results/gen.json}
SIF_IMAGE=${SIF_IMAGE:-ouro.sif}
unset https_proxy

# ── Paths ──
PROJECT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
HF_CACHE=${HF_HOME:-$HOME/.cache/huggingface}

mkdir -p "$PROJECT_DIR/results" slurm/logs

echo "=== Generation Job $SLURM_JOB_ID ==="
echo "Adapter:        ${ADAPTER_PATH:-(none — base model)}"
echo "ut_steps:       $TOTAL_UT_STEPS"
echo "Prompts:        $PROMPTS_FILE"
echo "max_new_tokens: $MAX_NEW_TOKENS"
echo "Output:         $OUTPUT_FILE"
echo "Node:           $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

module load singularity
singularity exec --nv --contain --writable-tmpfs \
    --bind "$PROJECT_DIR:/workspace" \
    --bind "$HF_CACHE:/scratch/huggingface" \
    "$SIF_IMAGE" \
    bash -c "cd /workspace && python eval_generation.py \
        --total_ut_steps $TOTAL_UT_STEPS \
        --prompts_file $PROMPTS_FILE \
        --max_new_tokens $MAX_NEW_TOKENS \
        --output_file $OUTPUT_FILE \
        ${ADAPTER_PATH:+--adapter_path $ADAPTER_PATH}"

echo "=== Generation Job $SLURM_JOB_ID finished ==="
