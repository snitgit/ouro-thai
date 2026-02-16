#!/bin/bash
# Submit all experiment runs from the experiment matrix
# Usage: bash slurm/run_experiments.sh

set -e

BASE_CONFIG=configs/a100.yaml

echo "Submitting experiment matrix..."

# Run 1: Main experiment — WangchanThaiInstruct, ut_steps=4
sbatch --export=ALL,CONFIG=$BASE_CONFIG,OUTPUT_DIR=./output/exp1-wangchan-ut4,TOTAL_UT_STEPS=4 \
    --job-name=ouro-ut4 \
    slurm/train.sbatch
echo "Submitted: exp1 — WangchanThaiInstruct ut_steps=4"

# Run 2: Baseline (no looping) — WangchanThaiInstruct, ut_steps=1
sbatch --export=ALL,CONFIG=$BASE_CONFIG,OUTPUT_DIR=./output/exp2-wangchan-ut1,TOTAL_UT_STEPS=1 \
    --job-name=ouro-ut1 \
    slurm/train.sbatch
echo "Submitted: exp2 — WangchanThaiInstruct ut_steps=1"

# Run 3a: Ablation — WangchanThaiInstruct, ut_steps=2
sbatch --export=ALL,CONFIG=$BASE_CONFIG,OUTPUT_DIR=./output/exp3a-wangchan-ut2,TOTAL_UT_STEPS=2 \
    --job-name=ouro-ut2 \
    slurm/train.sbatch
echo "Submitted: exp3a — WangchanThaiInstruct ut_steps=2"

# Run 3b: Ablation — WangchanThaiInstruct, ut_steps=3
sbatch --export=ALL,CONFIG=$BASE_CONFIG,OUTPUT_DIR=./output/exp3b-wangchan-ut3,TOTAL_UT_STEPS=3 \
    --job-name=ouro-ut3 \
    slurm/train.sbatch
echo "Submitted: exp3b — WangchanThaiInstruct ut_steps=3"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
