# Ouro Thai QLoRA Fine-Tuning — Progress Notes

## Project Goal
Fine-tune ByteDance/Ouro-2.6B (Looped Language Model) for Thai using QLoRA.
Research question: Does the looping mechanism help or hurt cross-lingual adaptation?

## Setup Progress

### Environment (Docker-based)
- [x] Docker installed and working on WSL2
- [x] NVIDIA GPU detected (CUDA 12.8, driver 560.94)
- [x] nvidia-container-toolkit installed and configured
- [x] Verified GPU passthrough works with vLLM container

### Pipeline Files Created
- [x] `Dockerfile` — CUDA 12.8.1 base, Ubuntu 24.04, all deps
- [x] `requirements.txt` — transformers==4.54.1, peft, bitsandbytes, trl, etc.
- [x] `train_qlora.py` — Main training script with configurable `total_ut_steps`
- [x] `data_config.py` — Dataset loading (WangchanThaiInstruct)
- [x] `docker-compose.yml` — GPU passthrough, HF cache mount, wandb config
- [x] `configs/default.yaml` — Default hyperparameters for RTX 3080
- [x] `run.sh` — Build & run helper

### Completed
- [x] Build Docker image (`docker compose build`)
- [x] Test with `--max_steps 10` — no OOM on RTX 3080 10GB
- [x] Verify WangchanThaiInstruct dataset loads correctly (32,207 train examples)
- [x] Verified LoRA applies correctly (30.3M trainable / 2.7B total = 1.12%)

### TODO
- [x] Add second RTX 3080 Ti — multi-GPU setup (verified, 22GB total)
- [ ] Increase batch_size to 2 (leverage dual GPU VRAM)
- [x] Run full training with ut_steps=4 (job 337867 — COMPLETED)
- [ ] Run experiments with ut_steps=1,2,3 for comparison (running sequentially)
- [ ] Optional: train on WangchanX Synthetic Instruct (120K) for comparison
- [ ] Optional: train on Thai-R1-Distill-SFT (10K) for reasoning experiment
- [ ] Evaluate on ThaiExam + OpenThaiEval + MT-Bench-Thai
- [ ] Benchmark against OpenThaiGPT / Typhoon / WangchanGLM

## Key Decisions
- **Base model**: Ouro-2.6B (not Thinking variant — cleaner for language adaptation)
- **Dataset**: airesearch/WangchanThaiInstruct (32,207 examples)
- **Quantization**: 4-bit NF4 with double quantization
- **LoRA targets**: All attention + MLP projections (q,k,v,o,gate,up,down)
- **VRAM budget**: ~7-8GB for 2.6B 4-bit + LoRA, fits RTX 3080 10GB

## Bugs Fixed During Setup
- Added `trust_remote_code=True` for Ouro custom model code
- Removed `max_seq_length` from SFTTrainer (not supported in current trl version)
- Cast `learning_rate` to `float()` — YAML reads `2e-4` as string
- Switched from `fp16` to `bf16` — fixes grad scaler assertion error with paged_adamw_8bit
- Added `--report_to` CLI arg for disabling wandb during testing
- Label masking fix: loss was computed on all tokens (including prompt), causing instant memorization (loss→0 by step 40). `DataCollatorForCompletionOnlyLM` doesn't exist in trl 0.20.0 — switched to `DataCollatorForChatML` from `trl.trainer.utils` instead. Refactored `data_config.py` to return raw `messages` column (not pre-formatted text)
- `DataCollatorForChatML` requires `SFTConfig` with `dataset_kwargs={"skip_prepare_dataset": True}` and `remove_unused_columns=False` — otherwise SFTTrainer tokenizes/strips the `messages` column before the collator can use it
- Added `module load singularity` to sbatch script — `singularity` not available by default on compute nodes
- WangchanThaiInstruct uses capitalized field names (`Instruction`, `Input`, `Output`) — mapper was using lowercase, producing empty messages
- HuggingFace datasets cache served stale results after fixing field names — added `load_from_cache_file=False` to force re-processing

## Test Run Results (10 steps)
- **Loss**: 3.93 (expected for start)
- **Speed**: ~19.7s/step
- **No OOM** on RTX 3080 10GB
- Adapter saved successfully

## Notes
- Must use `transformers<4.56.0` (4.54.1 recommended) for Ouro compatibility
- vLLM v0.6.6 works with CUDA 12.6 but didn't support Ouro architecture
- Docker base image uses CUDA 12.8.1 (Ubuntu 24.04), PyTorch 2.6.0+cu128
- Use `--report_to none` when running without wandb configured

## Dataset Options

### Training (SFT)
| Dataset | Size | Notes |
|---------|------|-------|
| airesearch/WangchanThaiInstruct | 32,207 | Current choice. Established, well-known |
| WangchanX Seed-Free Synthetic Instruct | 120,000 | ~4x larger, synthetic, by VISTEC |
| Thai-R1-Distill-SFT (iApp) | 10,000 | Distilled from reasoning models — good fit for testing Ouro's looping mechanism |

**Plan**: Use WangchanThaiInstruct as primary. WangchanX Synthetic (120K) as scale-up experiment. Thai-R1-Distill-SFT as reasoning-focused ablation — particularly interesting for the research question since Ouro's loop = latent reasoning.

### Evaluation Benchmarks
| Benchmark | Size | What it measures |
|-----------|------|-----------------|
| ThaiExam | — | Real Thai knowledge (ONET, TGAT, IC exams) |
| OpenThaiEval | 1,200 | General Thai NLP tasks |
| MT-Bench-Thai | — | Multi-turn conversational quality |

**Eval plan**: Run all three benchmarks across ut_steps=1,2,3,4 to show whether more loops improve Thai performance. Compare against OpenThaiGPT / Typhoon / WangchanGLM baselines.

## Experiment Matrix

| Run | Dataset | ut_steps | Purpose |
|-----|---------|----------|---------|
| 1 | WangchanThaiInstruct (32K) | 4 | Main experiment |
| 2 | WangchanThaiInstruct (32K) | 1 | Baseline (no looping) |
| 3 | WangchanThaiInstruct (32K) | 2,3 | Ablation on loop count |
| 4 | WangchanX Synthetic (120K) | 4 | Scale-up experiment |
| 5 | Thai-R1-Distill-SFT (10K) | 1 vs 4 | Does looping help reasoning in Thai? |

## Multi-GPU Setup (RTX 3080 + RTX 3080 Ti)
- Both GPUs verified working in Docker (22GB total VRAM)
- Docker GPU order: GPU 0 = 3080 Ti (12GB), GPU 1 = 3080 (10GB)
- `device_map="auto"` splits model across GPUs — works but adds ~2s/step overhead
- 10-step test with 2 GPUs: loss=3.92, ~21.7s/step (vs ~19.7s/step single GPU)
- **Next step**: increase batch_size to 2 to actually leverage the extra VRAM
- Consider: full bf16 LoRA (no 4-bit) for better training quality with 22GB headroom

## BCM Cluster Setup (Singularity + 4x H100)

### Hardware
- **Node**: tau (bcm-ai-h02)
- **GPUs**: 8x NVIDIA H100 80GB HBM3 (using 4 per job)
- **Partition**: defq

### New Files
- `singularity.def` — Container definition (torch 2.7.1, HF_HOME=/scratch/huggingface)
- `configs/a100.yaml` — Cluster config: no quantization (full bf16), batch_size=8, grad_accum=4
- `slurm/train.sbatch` — Slurm job script (partition=defq, 4 GPUs, 256GB RAM, 24h)
- `slurm/run_experiments.sh` — Submits all 4 experiment runs (ut_steps=1,2,3,4)

### Changes to train_qlora.py
- Added `use_qlora` config flag (default true for backward compat)
- When `use_qlora: false`: skips 4-bit quantization, loads full bf16, uses adamw_torch
- Added env var overrides: `CONFIG`, `OUTPUT_DIR`, `REPORT_TO`

### Cluster Workflow
```bash
# 1. Build Singularity image locally (WSL2 with apptainer)
apptainer build --fakeroot ouro.sif singularity.def

# 2. Upload project + .sif to cluster
scp -r ~/llm/ouro snit.san@bcm-ai-h02:~/llm/Ouro

# 3. Submit training job
sbatch --export=ALL,REPORT_TO=none slurm/train.sbatch

# 4. Submit all experiments
bash slurm/run_experiments.sh

# 5. Monitor
squeue -u $USER
tail -f slurm/logs/<jobid>_ouro-thai.err
sacct --format=JobID,JobName,State,ExitCode,Elapsed -u $USER
```

### Issues Resolved
- `singularity build --fakeroot` fails on cluster — build locally on WSL2 with apptainer instead
- Slurm spool mount permission error — fixed with `--contain --writable-tmpfs`
- Config file not found inside container — fixed with `bash -c "cd /workspace && ..."`
- `torch==2.6.0` no longer on PyPI for cu128 — updated to `torch==2.7.1`
- Cluster proxy interference — added `unset https_proxy` in sbatch
- `TRANSFORMERS_CACHE` deprecated warning — harmless, use `HF_HOME` in future

### First Cluster Run (job 337857) — FAILED
- **3,021 total steps** (32K × 3 epochs / batch 8 / grad_accum 4)
- **~5.7s/step** on 4x H100
- **Est. total time: ~4.8h**
- **Result**: Loss collapsed to 0.0 by step 40, mean_token_accuracy=1.0 — model memorized instead of learning
- **Root cause**: No label masking — loss was computed on ALL tokens (prompt + response), so model learned to predict the trivially repetitive ChatML template tokens
- **Fix applied**:
  - Switched to `DataCollatorForChatML` (from `trl.trainer.utils`) — `DataCollatorForCompletionOnlyLM` doesn't exist in trl 0.20.0
  - Refactored `data_config.py` to return raw `messages` column instead of pre-formatted text
  - `DataCollatorForChatML` handles tokenization + label masking: only computes loss on assistant response tokens

### Second Cluster Run (job 337862, 337863) — FAILED
- job 337862: `KeyError: 'messages'` — SFTTrainer tokenized and stripped `messages` column before collator could use it. Fix: `dataset_kwargs={"skip_prepare_dataset": True}`
- job 337863: `ValueError: No columns match model's forward method` — Trainer removed `messages` column. Fix: `remove_unused_columns=False`

### Third Cluster Run (job 337864, 337865) — FAILED
- Loss still collapsed to 0.0 — data was still empty despite field name fix
- Root cause: HF datasets cache served stale results from old mapper (lowercase field names)
- Fix: added `load_from_cache_file=False` to `ds.map()`

### Fourth Cluster Run (job 337867) — COMPLETED (exp1: ut_steps=4)
- All fixes verified: data contains real Thai text (1854/2048 response tokens for sample 0)
- **3,021 steps** completed in **~27.3h** (~31.5s/step on 4x H100)
- **Final loss**: 0.274 | **Avg train loss**: 0.324
- **Final token accuracy**: 91.2% | **Avg**: 90.4%
- **194M tokens** processed across 3 epochs
- Adapter saved to `/workspace/output/final_adapter`
- Training healthy — loss plateaued ~0.25-0.29 in final epoch, no overfitting

### Fifth Cluster Run (job 337944) — COMPLETED (exp2: ut_steps=1, baseline)
- Jobs 337942, 337943 failed (wandb auth — forgot `REPORT_TO=none`)
- **3,021 steps** completed in **~7.1h** (25,600s, ~8.5s/step on 4x H100)
- **Final loss**: 0.344 | **Avg train loss**: 0.437
- **Final token accuracy**: 88.9% | **Avg**: 88.1%
- **194M tokens** processed across 3 epochs
- Adapter saved to `./output/exp2-wangchan-ut1`

### Sixth Cluster Run (jobs 337950, 337951) — IN PROGRESS (exp3a: ut_steps=2, exp3b: ut_steps=3)
- job 337950: `mini-yut2` — ut_steps=2, output: `./output/exp3a-wangchan-ut2`
- job 337951: `mini-yut3` — ut_steps=3, output: `./output/exp3b-wangchan-ut3`
- Both running on `tau` with `REPORT_TO=none`

### Results Summary So Far
| Experiment | ut_steps | Final Loss | Avg Loss | Final Acc | Avg Acc | Runtime |
|---|---|---|---|---|---|---|
| exp1 (337867) | 4 | 0.274 | 0.324 | 91.2% | 90.4% | 27.3h |
| exp2 (337944) | 1 | 0.344 | 0.437 | 88.9% | 88.1% | 7.1h |
| exp3a (337950) | 2 | — | — | — | — | running |
| exp3b (337951) | 3 | — | — | — | — | running |

### Local vs Cluster Comparison
| | Local (2x RTX 3080) | Cluster (4x H100) |
|---|---|---|
| VRAM | 22GB total | 320GB total |
| Quantization | 4-bit NF4 (QLoRA) | None (full bf16 LoRA) |
| Batch size | 2 | 8 |
| Effective batch | 16 | 32 |
| Speed | ~22s/step | ~5.7s/step |
| Total steps | 12,078 | 3,021 |
| Est. time per run | ~76h | ~4.8h |
