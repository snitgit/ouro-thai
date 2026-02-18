"""eval.py — Thai MCQ evaluation for Ouro ut_steps ablation study.

Scores each model variant on multiple-choice questions using log-probability
of each answer token (A/B/C/D). More reliable than free-text generation.

Usage:
    # Fine-tuned model
    python eval.py \
        --adapter_path output/ouro-2.6b-thai-ut4/final_adapter \
        --total_ut_steps 4 \
        --output_file results/eval_ft_ut4.json

    # Base model (no adapter)
    python eval.py \
        --total_ut_steps 4 \
        --output_file results/eval_base.json
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


CHOICE_LABELS = ["A", "B", "C", "D", "E"]

# Column mappings for supported datasets
DATASET_CONFIGS = {
    # scb10x/thai_exam: subsets onet, tgat, a_level, ic, tpat1
    # columns: question, a, b, c, d, e (optional), answer (lowercase letters)
    # splits: train (5-shot, 5 rows), test (eval)
    "scb10x/thai_exam": {
        "question_col": "question",
        "choice_cols": ["a", "b", "c", "d", "e"],
        "answer_col": "answer",
        "choices_are_columns": True,
    },
}


def build_prompt(tokenizer, question, choices):
    """Build ChatML-formatted MCQ prompt. choices is a list of non-empty option texts."""
    labels = CHOICE_LABELS[:len(choices)]
    choices_text = "\n".join(f"{label}. {text}" for label, text in zip(labels, choices))
    label_list = ", ".join(labels[:-1]) + f" หรือ {labels[-1]}"
    user_content = (
        f"จงเลือกคำตอบที่ถูกต้องที่สุด\n\n"
        f"คำถาม: {question}\n\n"
        f"{choices_text}\n\n"
        f"ตอบด้วยตัวอักษร {label_list} เพียงตัวเดียว"
    )
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def score_choices(model, tokenizer, prompt, n_choices, device):
    """Return log-probs for the first n_choices option tokens after the prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs, use_cache=False)
    logits = outputs.logits[0, -1, :]  # last token position
    log_probs = torch.log_softmax(logits, dim=-1)

    scores = []
    for label in CHOICE_LABELS[:n_choices]:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        scores.append(log_probs[token_ids[0]].item() if token_ids else float("-inf"))
    return scores


def normalize_answer_to_index(answer):
    """Convert answer label (A/B/C/D, a/b/c/d) or int to 0-based index."""
    if isinstance(answer, str):
        answer = answer.strip().upper()
        if answer in CHOICE_LABELS:
            return CHOICE_LABELS.index(answer)
    return int(answer) if not isinstance(answer, int) else answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ByteDance/Ouro-2.6B")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter. Omit to evaluate base model.",
    )
    parser.add_argument("--total_ut_steps", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="scb10x/thai_exam")
    parser.add_argument(
        "--subset",
        type=str,
        default="onet",
        help="Dataset config/subset name (e.g. onet, tgat, ic).",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit to N samples (for debugging).",
    )
    args = parser.parse_args()

    label = f"ft_ut{args.total_ut_steps}" if args.adapter_path else "base"
    print(f"=== Ouro Eval: {label} ===")
    print(f"Model:     {args.model_name}")
    print(f"Adapter:   {args.adapter_path or '(none — base model)'}")
    print(f"ut_steps:  {args.total_ut_steps}")
    print(f"Dataset:   {args.dataset} / {args.subset} / {args.split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer from adapter dir (has chat template) or base model
    tok_source = args.adapter_path if args.adapter_path else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model (full bf16 — no quantization for eval)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        total_ut_steps=args.total_ut_steps,
        trust_remote_code=True,
    )

    if args.adapter_path:
        print("Loading adapter...")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    # Load dataset
    cfg = DATASET_CONFIGS.get(
        args.dataset,
        {
            "question_col": "question",
            "choice_cols": ["a", "b", "c", "d"],
            "answer_col": "answer",
            "choices_are_columns": True,
        },
    )

    print("Loading dataset...")
    ds = load_dataset(args.dataset, args.subset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"Evaluating on {len(ds)} examples")

    results = []
    correct = 0

    for i, example in enumerate(ds):
        question = example[cfg["question_col"]]

        if cfg.get("choices_are_columns"):
            # Filter out empty/None options (e.g. questions with only 4 of 5 cols)
            choices = [
                example[col] for col in cfg["choice_cols"]
                if example.get(col) not in (None, "")
            ]
        else:
            choices = [c for c in example[cfg["choices_col"]] if c not in (None, "")]

        correct_idx = normalize_answer_to_index(example[cfg["answer_col"]])

        prompt = build_prompt(tokenizer, question, choices)
        scores = score_choices(model, tokenizer, prompt, len(choices), device)
        pred_idx = int(np.argmax(scores))
        is_correct = pred_idx == correct_idx

        if is_correct:
            correct += 1

        results.append(
            {
                "id": i,
                "question": question,
                "choices": choices,
                "answer": CHOICE_LABELS[correct_idx],
                "predicted": CHOICE_LABELS[pred_idx],
                "scores": scores,
                "correct": is_correct,
            }
        )

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ds)}] acc={correct/(i+1):.3f}")

    accuracy = correct / len(ds)
    print(f"\nAccuracy: {accuracy:.4f}  ({correct}/{len(ds)})")

    output = {
        "model": args.model_name,
        "adapter": args.adapter_path,
        "total_ut_steps": args.total_ut_steps,
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "n_examples": len(ds),
        "n_correct": correct,
        "accuracy": accuracy,
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()
