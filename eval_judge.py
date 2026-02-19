"""eval_judge.py — LLM-as-judge scoring for Thai instruction-following eval.

Reads generation outputs from eval_generation.py and scores each response using
Claude as judge, following the MT-Bench LLM-as-judge methodology (Zheng et al., 2023).

Requires: pip install anthropic
          export ANTHROPIC_API_KEY=<your key>

Usage:
    python eval_judge.py \\
        --results_dir results/ \\
        --output_file results/judge_scores.json

    # Score only specific model variants
    python eval_judge.py \\
        --results_dir results/ \\
        --labels base ft_ut1 ft_ut4 \\
        --output_file results/judge_scores.json
"""

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

import anthropic

JUDGE_SYSTEM = (
    "คุณเป็นผู้ประเมินคุณภาพของโมเดลภาษาไทยที่เชี่ยวชาญ "
    "คุณมีหน้าที่ประเมินคำตอบของโมเดลภาษาอย่างตรงไปตรงมาและยุติธรรม "
    "ให้คะแนนตามเกณฑ์ที่กำหนดอย่างเคร่งครัด"
)

JUDGE_PROMPT = """\
กรุณาประเมินคำตอบต่อไปนี้สำหรับคำสั่ง/คำถามที่ให้มา

## คำสั่ง:
{instruction}

## คำตอบของโมเดล:
{response}

กรุณาให้คะแนน 1–10 ในแต่ละมิติ:
1. **Instruction Following** — ทำตามคำสั่งครบถ้วนและถูกต้องแค่ไหน (1=ไม่ทำตามเลย, 10=ทำตามครบทุกอย่าง)
2. **Thai Language Quality** — ภาษาไทยถูกต้อง เป็นธรรมชาติ อ่านเข้าใจง่ายแค่ไหน (1=ผิดมาก ไม่เป็นธรรมชาติ, 10=ถูกต้องสมบูรณ์)
3. **Helpfulness** — คำตอบมีประโยชน์และมีคุณค่าแค่ไหน (1=ไม่มีประโยชน์, 10=มีประโยชน์มากที่สุด)

ตอบเป็น JSON เท่านั้น ไม่ต้องมีคำอธิบายเพิ่มเติมนอก JSON:
{{"instruction_following": <int 1-10>, "thai_quality": <int 1-10>, "helpfulness": <int 1-10>, "comment": "<brief Thai comment>"}}"""


def score_response(client, instruction, response, model):
    """Call Claude to score a single response. Returns dict with scores."""
    prompt = JUDGE_PROMPT.format(instruction=instruction, response=response)
    message = client.messages.create(
        model=model,
        max_tokens=256,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text.strip()

    # Handle cases where the model wraps JSON in ```json ... ```
    if "```" in text:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        text = match.group(0) if match else text

    scores = json.loads(text)
    scores["total"] = (
        scores["instruction_following"]
        + scores["thai_quality"]
        + scores["helpfulness"]
    )
    return scores


def load_gen_files(results_dir, labels_filter):
    """Load gen_*.json files. Returns {label: {prompt_id: item}}."""
    results_path = Path(results_dir)
    gen_files = sorted(results_path.glob("gen_*.json"))
    if not gen_files:
        raise FileNotFoundError(f"No gen_*.json files found in {results_dir}")

    all_gens = {}
    for fpath in gen_files:
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        label = data["label"]
        if labels_filter and label not in labels_filter:
            continue
        all_gens[label] = {r["id"]: r for r in data["results"]}
        print(f"  Loaded {len(data['results'])} results for '{label}' ({fpath.name})")
    return all_gens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/",
        help="Directory containing gen_*.json files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/judge_scores.json",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="claude-sonnet-4-6",
        help="Claude model to use as judge.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Restrict scoring to specific model labels (e.g. base ft_ut1 ft_ut4).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Seconds to wait between API calls.",
    )
    args = parser.parse_args()

    print(f"=== Ouro LLM-as-Judge Eval ===")
    print(f"Judge model: {args.judge_model}")
    print(f"Results dir: {args.results_dir}")

    all_gens = load_gen_files(args.results_dir, set(args.labels) if args.labels else None)
    if not all_gens:
        print("No matching generation files found.")
        return

    labels = list(all_gens.keys())
    first_label = labels[0]
    prompt_ids = list(all_gens[first_label].keys())
    print(f"\nModels: {labels}")
    print(f"Prompts: {len(prompt_ids)}")
    print(f"Total API calls: {len(prompt_ids) * len(labels)}\n")

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    scores_by_prompt = {}
    call_count = 0
    total_calls = len(prompt_ids) * len(labels)

    for prompt_id in prompt_ids:
        scores_by_prompt[prompt_id] = {}
        for label in labels:
            if prompt_id not in all_gens[label]:
                continue
            item = all_gens[label][prompt_id]
            call_count += 1
            print(f"  [{call_count}/{total_calls}] {prompt_id} / {label}")
            try:
                scores = score_response(
                    client, item["instruction"], item["response"], args.judge_model
                )
                scores_by_prompt[prompt_id][label] = scores
                print(
                    f"    IF={scores['instruction_following']} "
                    f"TQ={scores['thai_quality']} "
                    f"H={scores['helpfulness']} "
                    f"Total={scores['total']}  — {scores.get('comment', '')}"
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                scores_by_prompt[prompt_id][label] = {"error": str(e)}

            if args.delay > 0:
                time.sleep(args.delay)

    # Aggregate scores per model
    metrics = ["instruction_following", "thai_quality", "helpfulness", "total"]
    model_vals = {label: {m: [] for m in metrics} for label in labels}
    for prompt_id, model_scores in scores_by_prompt.items():
        for label, scores in model_scores.items():
            if "error" in scores:
                continue
            for m in metrics:
                model_vals[label][m].append(scores[m])

    summary = {}
    for label in labels:
        vals = model_vals[label]
        if not vals["total"]:
            continue
        summary[label] = {m: round(statistics.mean(vals[m]), 2) for m in metrics}

    # Print summary table
    print("\n=== Summary (mean scores, max 10 per dimension, 30 total) ===")
    print(f"{'Model':<20} {'IF':>6} {'TQ':>6} {'Help':>6} {'Total':>7}")
    print("-" * 50)
    for label in sorted(summary.keys()):
        s = summary[label]
        print(
            f"{label:<20} {s['instruction_following']:>6.2f} "
            f"{s['thai_quality']:>6.2f} {s['helpfulness']:>6.2f} "
            f"{s['total']:>7.2f}"
        )

    output = {
        "judge_model": args.judge_model,
        "labels": labels,
        "n_prompts": len(prompt_ids),
        "scores_by_prompt": scores_by_prompt,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {args.output_file}")


if __name__ == "__main__":
    main()
