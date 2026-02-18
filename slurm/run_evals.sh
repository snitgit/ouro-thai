#!/bin/bash
# Submit eval jobs for all ut_steps variants + base model.
# Usage: bash slurm/run_evals.sh [dataset] [subset] [split]
#
# Adapter paths (in-container /workspace/output/...):
#   base  — no adapter, ut_steps=4
#   ut1   — exp2-wangchan-ut1/final_adapter
#   ut2   — exp3a-wangchan-ut2/final_adapter
#   ut3   — exp3b-wangchan-ut3/final_adapter
#   ut4   — ouro-2.6b-thai-ut4/final_adapter

set -e

DATASET=${1:-scb10x/thai_exam}
SUBSET=${2:-onet}
SPLIT=${3:-test}

echo "Dataset: $DATASET / $SUBSET / $SPLIT"
echo "Submitting eval jobs..."

# Base model (no adapter, ut_steps=4 = full model capacity)
sbatch \
    --job-name=eval-base \
    --export=ALL,\
TOTAL_UT_STEPS=4,\
DATASET=$DATASET,\
SUBSET=$SUBSET,\
SPLIT=$SPLIT,\
OUTPUT_FILE=/workspace/results/eval_base.json \
    slurm/eval.sbatch
echo "Submitted: base (no adapter, ut_steps=4)"

# Fine-tuned ut_steps=1
sbatch \
    --job-name=eval-ft-ut1 \
    --export=ALL,\
ADAPTER_PATH=/workspace/output/exp2-wangchan-ut1/final_adapter,\
TOTAL_UT_STEPS=1,\
DATASET=$DATASET,\
SUBSET=$SUBSET,\
SPLIT=$SPLIT,\
OUTPUT_FILE=/workspace/results/eval_ft_ut1.json \
    slurm/eval.sbatch
echo "Submitted: ft-ut1"

# Fine-tuned ut_steps=2
sbatch \
    --job-name=eval-ft-ut2 \
    --export=ALL,\
ADAPTER_PATH=/workspace/output/exp3a-wangchan-ut2/final_adapter,\
TOTAL_UT_STEPS=2,\
DATASET=$DATASET,\
SUBSET=$SUBSET,\
SPLIT=$SPLIT,\
OUTPUT_FILE=/workspace/results/eval_ft_ut2.json \
    slurm/eval.sbatch
echo "Submitted: ft-ut2"

# Fine-tuned ut_steps=3
sbatch \
    --job-name=eval-ft-ut3 \
    --export=ALL,\
ADAPTER_PATH=/workspace/output/exp3b-wangchan-ut3/final_adapter,\
TOTAL_UT_STEPS=3,\
DATASET=$DATASET,\
SUBSET=$SUBSET,\
SPLIT=$SPLIT,\
OUTPUT_FILE=/workspace/results/eval_ft_ut3.json \
    slurm/eval.sbatch
echo "Submitted: ft-ut3"

# Fine-tuned ut_steps=4
sbatch \
    --job-name=eval-ft-ut4 \
    --export=ALL,\
ADAPTER_PATH=/workspace/output/ouro-2.6b-thai-ut4/final_adapter,\
TOTAL_UT_STEPS=4,\
DATASET=$DATASET,\
SUBSET=$SUBSET,\
SPLIT=$SPLIT,\
OUTPUT_FILE=/workspace/results/eval_ft_ut4.json \
    slurm/eval.sbatch
echo "Submitted: ft-ut4"

echo ""
echo "All 5 eval jobs submitted. Monitor with: squeue -u \$USER"
echo "Results will appear in: results/"
