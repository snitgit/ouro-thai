#!/bin/bash
# Submit perplexity eval jobs for all ut_steps variants + base model.
# Dataset: Thai Wikipedia (wikimedia/wikipedia, 20231101.th), 500 articles.
# Usage: bash slurm/run_ppl_evals.sh [max_samples]

set -e

MAX_SAMPLES=${1:-500}

echo "Submitting perplexity eval jobs (max_samples=$MAX_SAMPLES)..."

# Base model
sbatch \
    --job-name=mini-ep0 \
    --export=ALL,\
TOTAL_UT_STEPS=4,\
MAX_SAMPLES=$MAX_SAMPLES,\
OUTPUT_FILE=/workspace/results/ppl_base.json \
    slurm/ppl.sbatch
echo "Submitted: ppl-base"

# Fine-tuned ut_steps=1
sbatch \
    --job-name=mini-ep1 \
    --export=ALL,\
ADAPTER_PATH=/workspace/output/exp2-wangchan-ut1/final_adapter,\
TOTAL_UT_STEPS=1,\
MAX_SAMPLES=$MAX_SAMPLES,\
OUTPUT_FILE=/workspace/results/ppl_ft_ut1.json \
    slurm/ppl.sbatch
echo "Submitted: ppl-ft-ut1"

# Fine-tuned ut_steps=2
sbatch \
    --job-name=mini-ep2 \
    --export=ALL,\
ADAPTER_PATH=/workspace/output/exp3a-wangchan-ut2/final_adapter,\
TOTAL_UT_STEPS=2,\
MAX_SAMPLES=$MAX_SAMPLES,\
OUTPUT_FILE=/workspace/results/ppl_ft_ut2.json \
    slurm/ppl.sbatch
echo "Submitted: ppl-ft-ut2"

# Fine-tuned ut_steps=3
sbatch \
    --job-name=mini-ep3 \
    --export=ALL,\
ADAPTER_PATH=/workspace/output/exp3b-wangchan-ut3/final_adapter,\
TOTAL_UT_STEPS=3,\
MAX_SAMPLES=$MAX_SAMPLES,\
OUTPUT_FILE=/workspace/results/ppl_ft_ut3.json \
    slurm/ppl.sbatch
echo "Submitted: ppl-ft-ut3"

# Fine-tuned ut_steps=4
sbatch \
    --job-name=mini-ep4 \
    --export=ALL,\
ADAPTER_PATH=/workspace/output/ouro-2.6b-thai-ut4/final_adapter,\
TOTAL_UT_STEPS=4,\
MAX_SAMPLES=$MAX_SAMPLES,\
OUTPUT_FILE=/workspace/results/ppl_ft_ut4.json \
    slurm/ppl.sbatch
echo "Submitted: ppl-ft-ut4"

echo ""
echo "All 5 PPL jobs submitted. Monitor with: squeue -u \$USER"
echo "Results will appear in: results/ppl_*.json"
