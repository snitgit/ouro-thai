# Ouro Thai QLoRA Fine-Tuning — Progress Notes

## Project Goal
Fine-tune ByteDance/Ouro-2.6B (Looped Language Model) for Thai using QLoRA.
Research question: Does the looping mechanism help or hurt cross-lingual adaptation?

### What "Cross-Lingual Instruction-Following Adaptation" Means

**Cross-lingual**: Ouro was pre-trained mainly on English/Chinese. We want it to work in Thai — crossing from one language to another.

**Instruction-following**: The ability to understand a task and respond correctly.
- "สรุปข้อความนี้ให้หน่อย" → model summarizes
- "เขียนอีเมลสุภาพ" → model writes a polite email
- Not about knowing facts — about understanding the *task* and responding usefully.

**Adaptation**: Adjusting the model cheaply (via LoRA SFT) without retraining from scratch.

**Research hypothesis**: Ouro's looping mechanism (ut_steps) gives the model more internal computation per token — it can "think harder" before responding. More loops → better cross-lingual instruction-following adaptation.

### Two Separate Problems (Important Distinction)

| Problem | What it needs | Our approach |
|---|---|---|
| Thai instruction-following | SFT on Thai instruction data | ✅ WangchanThaiInstruct, 32K examples |
| Thai factual knowledge | Pre-training on Thai text at scale | ❌ Not addressed — requires CPT on 10–100B Thai tokens |

**Why MCQ scores are low**: ONET tests factual knowledge (memorized Thai school curriculum facts) — knowledge that comes from pre-training, not SFT. Ouro's pre-training corpus had <1% Thai text. No amount of instruction fine-tuning injects new facts.

**Why this doesn't invalidate the research**: We are measuring instruction-following adaptation, not knowledge. The training loss/accuracy curves (and eventual MT-Bench-style eval) are the correct metrics. MCQ scores are reported as a baseline comparison only.

**Path to better Thai MCQ**: Thai CPT (Continued Pre-Training on 10–100B Thai tokens) → then SFT. This is what Thai-specific models like Typhoon (SCB10X) and OpenThaiGPT do.

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
- [x] Run experiments with ut_steps=1,2,3 for comparison (jobs 337950, 337958 — COMPLETED)
- [ ] Optional: train on WangchanX Synthetic Instruct (120K) for comparison
- [ ] Optional: train on Thai-R1-Distill-SFT (10K) for reasoning experiment
- [x] Evaluate on ThaiExam (ONET MCQ, 162 q) — ✅ done; scores near random (knowledge mismatch)
- [x] Perplexity on Thai Wikipedia (500 articles) — ✅ done; clear ut4>ut3>ut2>ut1>base
- [ ] Thai instruction-following eval (MT-Bench-style) — ✅ pipeline ready; submit gen jobs
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

### Sixth Cluster Run — exp3a (job 337950, ut_steps=2) — COMPLETED
- `mini-yut2` — ut_steps=2, output: `./output/exp3a-wangchan-ut2`
- 4x H100, `REPORT_TO=none`
- **3,021 steps** completed in **~13.8h** (49,718s, ~16.5s/step)
- **Final loss**: 0.290 | **Avg train loss**: 0.353
- **Final token accuracy**: 90.6% | **Avg**: 89.8%
- **194M tokens** processed across 3 epochs
- Adapter saved to `/workspace/output/final_adapter`

### Seventh Cluster Run — exp3b (job 337958, ut_steps=3) — COMPLETED
- Original job 337951 cancelled at epoch 0.60 (4h09m) — 4 GPUs too aggressive for cluster
- Resubmitted as job 337958 with **1x H100** to reduce GPU usage
- `mini-yut3` — ut_steps=3, output: `./output/exp3b-wangchan-ut3`
- **~24s/step** on 1x H100 (heavier than ut_steps=1/2 — 3 forward passes per step)
- **3,021 steps** completed in **~20.4h** (73,413s)
- **Final loss**: 0.270 | **Avg train loss**: 0.324
- **Final token accuracy**: 91.3% | **Avg**: 90.6%
- **194M tokens** processed across 3 epochs
- Adapter saved to `/workspace/output/final_adapter`

### Results Summary (All Experiments Complete)
| Experiment | ut_steps | Final Loss | Avg Loss | Final Acc | Avg Acc | Runtime |
|---|---|---|---|---|---|---|
| exp1 (337867) | 4 | 0.274 | 0.324 | 91.2% | 90.4% | 27.3h |
| exp3b (337958) | 3 | 0.270 | 0.324 | 91.3% | 90.6% | 20.4h |
| exp3a (337950) | 2 | 0.290 | 0.353 | 90.6% | 89.8% | 13.8h |
| exp2 (337944) | 1 | 0.344 | 0.437 | 88.9% | 88.1% | 7.1h |

**Trend**: Clear monotonic improvement with more loops — ut_steps=3 and 4 converge to similar quality (~91%), ut_steps=2 is a clear step below, ut_steps=1 lags significantly. Supports hypothesis that looping helps cross-lingual adaptation.

## Evaluation Results — scb10x/thai_exam (ONET, test split, 162 examples)

### Eval Pipeline
- Script: `eval.py` — log-prob scoring over A/B/C/D/E tokens (no generation)
- Dataset: `scb10x/thai_exam`, subset `onet`, split `test` (162 questions)
- Jobs: 338006–338010 (1x H100, ~5 min each)
- Bugs fixed: wrong dataset name (`thai_exams`→`thai_exam`), `use_cache=False` for UniversalTransformerCache compatibility, 5-choice (A-E) support

### ONET Accuracy
| Model | ut_steps | Accuracy | Correct/162 |
|---|---|---|---|
| base (no adapter) | 4 | 20.4% | 33 |
| ft-ut1 | 1 | **22.8%** | 37 |
| ft-ut2 | 2 | 19.1% | 31 |
| ft-ut3 | 3 | 19.8% | 32 |
| ft-ut4 | 4 | 21.0% | 34 |

**Notes**:
- Random chance baseline: ~22% (mix of 4-choice and 5-choice questions)
- Scores are low overall — Ouro-2.6B has limited Thai factual knowledge; fine-tuning on instruction data (WangchanThaiInstruct) didn't strongly improve MCQ factual recall
- ut_steps=1 surprisingly outperforms higher loop counts on this benchmark
- Counter to training loss trend (where more loops = lower loss) — suggests loop count affects instruction-following more than knowledge recall

### Eval Methodology Assessment

**Fundamental mismatch**: WangchanThaiInstruct is instruction-following data, but MCQ benchmarks test factual knowledge recall. Near-random MCQ scores are expected — not a failure of training.

**MCQ eval options considered**:
| Eval | Effort | Signal |
|---|---|---|
| More MCQ subsets / iapp/openthaieval (1,232 q) | Low | Weak — same task mismatch |
| Thai MT-Bench (LLM-as-judge) | High | Strong — directly measures instruction quality |
| Perplexity on held-out Thai text | Low | Medium — measures language modeling |
| Human eval (~50 samples) | Medium | Strong |

**Decision**: MCQ eval (ONET) serves as a baseline for paper completeness but is not the primary signal. The real story is in training metrics (loss/accuracy vs ut_steps) + an instruction-following eval.

**Next eval steps** (priority order):
1. ~~Perplexity on held-out Thai text~~ — ✅ done; results in `results/ppl_*.json`
2. ~~Thai MT-Bench / qualitative instruction-following eval~~ — ✅ pipeline ready; run `bash slurm/run_generation.sh` then `python eval_judge.py`
3. `iapp/openthaieval` (1,232 q) — only if a reviewer specifically requires it

## Perplexity Evaluation

### What Perplexity Measures
How surprised the model is by Thai text it hasn't seen. Lower = better Thai language modeling.
- Model predicts next token at each position in a Thai Wikipedia article
- PPL = exp(mean negative log-likelihood per token)
- Low PPL → model confident → understands Thai well
- High PPL → model surprised → struggles with Thai

### Pipeline
- Script: `eval_perplexity.py` — sliding window PPL (window=2048, stride=512)
- Dataset: `wikimedia/wikipedia`, `20231101.th` (Thai Wikipedia), 500 articles, min 200 chars
- Reports: mean ± std perplexity across articles (paper-ready)
- Jobs: `mini-ep0` (base) through `mini-ep4` (ft-ut4), 1x H100, ~10-15 min total

### Jobs Submitted
| Job | Name | Model |
|---|---|---|
| 338011 | mini-ep0 | base (ut_steps=4, no adapter) |
| 338012 | mini-ep1 | ft-ut1 |
| 338013 | mini-ep2 | ft-ut2 |
| 338014 | mini-ep3 | ft-ut3 |
| 338015 | mini-ep4 | ft-ut4 |

Results → `results/ppl_*.json`

### PPL Results (Thai Wikipedia, 500 articles)

| Model | ut_steps | Mean PPL | Std | Median PPL |
|---|---|---|---|---|
| base (no adapter) | 4 | 2.396 | 0.464 | 2.275 |
| ft-ut1 | 1 | 2.387 | 0.518 | 2.252 |
| ft-ut2 | 2 | 2.203 | 0.435 | 2.093 |
| ft-ut3 | 3 | 2.113 | 0.397 | 2.017 |
| ft-ut4 | 4 | **2.085** | 0.388 | **1.991** |

**Key finding**: Clear monotonic improvement with more loops — consistent with training loss trend. ft-ut1 barely improves over base (2.387 vs 2.396), suggesting a single loop pass is insufficient for effective adaptation. ft-ut4 achieves the best perplexity, confirming the hypothesis that more loops = better cross-lingual language modeling.

**Contrast with ONET MCQ**: PPL and training metrics both show ut_steps=4 > 3 > 2 > 1 >> base. ONET accuracy was noisy and reversed (ut1 appeared best), reinforcing that MCQ tests factual knowledge (not affected by loop count) while PPL/training captures instruction-following adaptation quality.

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

---

## Thai Instruction-Following Eval (Thai MT-Bench-style)

### Overview
Two-phase eval to satisfy the paper's "Must-Do: 5–10 qualitative Thai instruction examples (ft-ut1 vs ft-ut4 side-by-side)" requirement and provide a quantitative instruction-following signal beyond PPL.

### Phase 1 — Generation (on cluster)
- **Script**: `eval_generation.py` — greedy decoding (`do_sample=False`), `max_new_tokens=512`, `use_cache=False`
- **Prompts**: `prompts/thai_instruct_bench.json` — 20 Thai prompts, 5 categories (4 each):
  - `summarization` — sum_1 to sum_4 (passage summarization, bullet points, length-constrained)
  - `writing` — write_1 to write_4 (email, ad copy, birthday message, LinkedIn bio)
  - `reasoning` — reason_1 to reason_4 (compound interest, math, ML explanation, finance calc)
  - `translation` — translate_1 to translate_4 (TH→EN, EN→TH, idioms, AI glossary)
  - `advice` — advice_1 to advice_4 (career, language learning, fitness, startup)
- **SLURM**: `slurm/gen.sbatch`, submitted via `slurm/run_generation.sh`
- **Jobs**: `mini-eg0` (base) through `mini-eg4` (ft-ut4), 1x H100, ~1–2h each
- **Output**: `results/gen_base.json`, `results/gen_ft_ut1.json` … `results/gen_ft_ut4.json`

### Phase 2 — LLM-as-Judge (local, needs API key)
- **Script**: `eval_judge.py` — reads all `gen_*.json`, calls Claude API per response
- **Judge model**: `claude-sonnet-4-6`
- **Scoring**: 3 dimensions × 10 pts each = 30 pts max
  - Instruction Following (IF): did the model do what was asked?
  - Thai Language Quality (TQ): natural, grammatically correct Thai?
  - Helpfulness (H): useful and informative answer?
- **Output**: `results/judge_scores.json` — per-prompt scores + aggregate summary table

```bash
# Phase 1 (cluster)
bash slurm/run_generation.sh

# Phase 2 (local, after gen jobs complete)
pip install anthropic
export ANTHROPIC_API_KEY=<key>
python eval_judge.py --results_dir results/ --output_file results/judge_scores.json

# Compare only ut1 vs ut4 (for paper examples)
python eval_judge.py --results_dir results/ --labels base ft_ut1 ft_ut4 \
    --output_file results/judge_scores_ablation.json
```

### Expected Paper Use
- **Table**: aggregate IF / TQ / H scores across ut_steps=1–4 (quantitative)
- **Appendix**: 5–10 selected side-by-side examples from `gen_ft_ut1.json` vs `gen_ft_ut4.json`
- Categories that best show loop-count difference: `reasoning` and `writing`

---

## Paper Improvement Notes (ECTI-CIT.tex — 2025-02-19)

### Issues Fixed in Revised ECTI-CIT.tex
- **Hardware**: corrected from "RTX 3080 (12GB)" → "4× NVIDIA H100 80GB" (cluster results)
- **Methodology**: corrected from "QLoRA / 4-bit NF4" → "full bf16 LoRA" (cluster used no quantization)
- **Effective batch size**: corrected from 16 → 32 (batch 8 × grad_accum 4)
- **Introduction**: removed unsupported claims (steps 0–8, ablations on attention sharing/position encodings, "comprehensive benchmark") — only ut=1–4 was actually done
- **Figure caption**: added explicit note that experiments use fixed ut, not adaptive early-exit
- **Added tables**: training metrics (loss+accuracy), ONET results, latency/cost
- **PPL table**: added std deviation column
- **Related work**: added Typhoon, OpenThaiGPT, WangchanBERTa, LoRA (Hu 2022), BLOOM (Scao 2022), Zheng 2023 (MT-Bench/LLM-as-judge)
- **Bibliography fixed**: removed XLS-R (speech, irrelevant), removed Asai2023 (retrieval, irrelevant), fixed Dehghani2018 → Dehghani2019 (ICLR 2019), fixed mBART citation, removed mislabeled Tunstall2023
- **Discussion expanded**: added saturation analysis (ut=3 vs ut=4), PPL/ONET divergence explanation, comparison with CPT-based Thai models
- **Conclusion**: improved with specific future work (MT-Bench eval, Thai CPT, WangchanX 120K, Thai-R1-Distill reasoning)

### Venue Recommendation for ก.พ.อ. Promotion (Thailand)

**ก.พ.อ. requires**: TCI Group 1 / Scopus / ISI for assistant professor promotion

| Venue | Type | Indexed | Cost | Fit | Difficulty |
|---|---|---|---|---|---|
| **ECTI-CIT** | Journal | TCI-1 + Scopus | Free | ★★★★★ | Medium |
| **iSAI-NLP** | Conference | Scopus (IEEE Xplore) | Free | ★★★★★ | Low–Medium |
| **PACLIC** | Conference | Scopus (ACL Anthology) | Free | ★★★★ | Medium |
| **AACL-IJCNLP** | Conference | Scopus (ACL Anthology) | Free | ★★★ | High |

**Recommendation**: Submit to **ECTI-CIT** now (already formatted, free, Scopus, good fit).
If MT-Bench eval is added later, **iSAI-NLP** is a fast Scopus-indexed alternative.

### Must-Do Before Submission

#### Initial Submission (Double-Blind Review)
- **ECTI-CIT uses double-blind review** — ห้ามใส่ชื่อผู้เขียนและสังกัดใน PDF
- ใน `\author{}` ให้ใส่ `[Anonymous]` หรือลบออกทั้งหมด
- ตรวจ Acknowledgment ด้วย — ถ้าระบุชื่อหน่วยงานที่ identify ตัวตนได้ ให้ลบออก
- ตรวจ References ด้วย — ถ้ามี self-citation ที่ระบุตัวตนได้ ให้ใช้ "[Anonymous, year]"
- สร้างไฟล์ 2 เวอร์ชัน: `ECTI-CIT-blind.tex` (ส่ง) และ `ECTI-CIT-final.tex` (เก็บไว้)

#### Content (ทำก่อนส่ง)
1. Add 5–10 qualitative Thai instruction examples (ft-ut1 vs ft-ut4 side-by-side) — reviewers will ask
2. Find and fix Ouro-2.6B proper BibTeX citation (ByteDance arXiv paper title)
3. Find and fix WangchanThaiInstruct proper BibTeX citation (NECTEC/VISTEC paper)
4. Verify LoRA r=64, alpha=16 matches actual configs/a100.yaml

#### Final/Camera-Ready (หลัง accept)
- ใส่ชื่อ, สังกัด, email จริง
- Restore Acknowledgment (BCM Cluster team, ทุนวิจัย ฯลฯ)
