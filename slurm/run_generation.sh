#!/bin/bash
# Submit generation eval jobs for all ut_steps variants + base model.
# Dataset: prompts/thai_instruct_bench.json (20 Thai instruction prompts, 5 categories).
# Usage: bash slurm/run_generation.sh [max_new_tokens]

set -e

MAX_NEW_TOKENS=${1:-512}

echo "Submitting generation eval jobs (max_new_tokens=$MAX_NEW_TOKENS)..."

# Base model
sbatch \
    --job-name=mini-eg0 \
    --export=ALL,\
TOTAL_UT_STEPS=4,\
MAX_NEW_TOKENS=$MAX_NEW_TOKENS,\
OUTPUT_FILE=/workspace/results/gen_base.json \
    slurm/gen.sbatch
echo "Submitted: gen-base"

# Fine-tuned ut_steps=1
sbatch \
    --job-name=mini-eg1 \
    --export=ALL,\
ADAPTER_PATH=/workspace/output/exp2-wangchan-ut1/final_adapter,\
TOTAL_UT_STEPS=1,\
MAX_NEW_TOKENS=$MAX_NEW_TOKENS,\
OUTPUT_FILE=/workspace/results/gen_ft_ut1.json \
    slurm/gen.sbatch
echo "Submitted: gen-ft-ut1"

# Fine-tuned ut_steps=2
sbatch \
    --job-name=mini-eg2 \
    --export=ALL,\
ADAPTER_PATH=/workspace/output/exp3a-wangchan-ut2/final_adapter,\
TOTAL_UT_STEPS=2,\
MAX_NEW_TOKENS=$MAX_NEW_TOKENS,\
OUTPUT_FILE=/workspace/results/gen_ft_ut2.json \
    slurm/gen.sbatch
echo "Submitted: gen-ft-ut2"

# Fine-tuned ut_steps=3
sbatch \
    --job-name=mini-eg3 \
    --export=ALL,\
ADAPTER_PATH=/workspace/output/exp3b-wangchan-ut3/final_adapter,\
TOTAL_UT_STEPS=3,\
MAX_NEW_TOKENS=$MAX_NEW_TOKENS,\
OUTPUT_FILE=/workspace/results/gen_ft_ut3.json \
    slurm/gen.sbatch
echo "Submitted: gen-ft-ut3"

# Fine-tuned ut_steps=4
sbatch \
    --job-name=mini-eg4 \
    --export=ALL,\
ADAPTER_PATH=/workspace/output/ouro-2.6b-thai-ut4/final_adapter,\
TOTAL_UT_STEPS=4,\
MAX_NEW_TOKENS=$MAX_NEW_TOKENS,\
OUTPUT_FILE=/workspace/results/gen_ft_ut4.json \
    slurm/gen.sbatch
echo "Submitted: gen-ft-ut4"

echo ""
echo "All 5 generation jobs submitted. Monitor with: squeue -u \$USER"
echo "Results will appear in: results/gen_*.json"
echo ""
echo "After all jobs complete, run the judge locally:"
echo "  pip install anthropic"
echo "  export ANTHROPIC_API_KEY=<your key>"
echo "  python eval_judge.py --results_dir results/ --output_file results/judge_scores.json"
