"""eval_perplexity.py — Thai language modeling perplexity for Ouro ut_steps ablation.

Computes perplexity on Thai Wikipedia using the sliding window method
(Press et al., 2021) to handle texts longer than the context window.

Dataset: wikimedia/wikipedia, 20231101.th (Thai Wikipedia)
Metric:  PPL = exp(mean NLL per token), reported as mean ± std across articles.

Usage:
    # Fine-tuned model
    python eval_perplexity.py \
        --adapter_path output/ouro-2.6b-thai-ut4/final_adapter \
        --total_ut_steps 4 \
        --output_file results/ppl_ft_ut4.json

    # Base model
    python eval_perplexity.py \
        --total_ut_steps 4 \
        --output_file results/ppl_base.json
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_perplexity(model, tokenizer, text, max_length, stride, device):
    """Sliding window perplexity (handles texts longer than context window).

    Only computes loss on the new tokens in each window (not the context prefix),
    giving the same result as full-sequence perplexity.
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings.input_ids[0]
    seq_len = input_ids.size(0)

    if seq_len < 2:
        return None

    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end  # tokens we're scoring in this window

        chunk = input_ids[begin:end].unsqueeze(0).to(device)
        target = chunk.clone()
        target[:, :-trg_len] = -100  # mask context prefix

        with torch.no_grad():
            loss = model(chunk, labels=target, use_cache=False).loss

        total_nll += loss.item() * trg_len
        total_tokens += trg_len
        prev_end = end

        if end == seq_len:
            break

    return float(np.exp(total_nll / total_tokens))


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
    parser.add_argument(
        "--dataset", type=str, default="wikimedia/wikipedia",
        help="HuggingFace dataset for perplexity eval.",
    )
    parser.add_argument(
        "--dataset_config", type=str, default="20231101.th",
        help="Dataset config/language (e.g. 20231101.th for Thai Wikipedia).",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--max_samples", type=int, default=500,
        help="Number of articles to evaluate.",
    )
    parser.add_argument(
        "--min_chars", type=int, default=200,
        help="Minimum article length in characters (filter stubs).",
    )
    parser.add_argument(
        "--max_length", type=int, default=2048,
        help="Context window size (tokens).",
    )
    parser.add_argument(
        "--stride", type=int, default=512,
        help="Sliding window stride (tokens).",
    )
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    label = f"ft_ut{args.total_ut_steps}" if args.adapter_path else "base"
    print(f"=== Ouro Perplexity Eval: {label} ===")
    print(f"Model:    {args.model_name}")
    print(f"Adapter:  {args.adapter_path or '(none — base model)'}")
    print(f"ut_steps: {args.total_ut_steps}")
    print(f"Dataset:  {args.dataset} / {args.dataset_config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tok_source = args.adapter_path if args.adapter_path else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (full bf16)
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
    print("Loading dataset...")
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split,
                      trust_remote_code=True)

    # Filter stubs and take max_samples
    ds = ds.filter(lambda x: len(x["text"]) >= args.min_chars)
    if len(ds) > args.max_samples:
        ds = ds.select(range(args.max_samples))
    print(f"Evaluating on {len(ds)} articles")

    ppls = []
    results = []

    for i, example in enumerate(ds):
        text = example["text"]
        ppl = compute_perplexity(
            model, tokenizer, text,
            max_length=args.max_length,
            stride=args.stride,
            device=device,
        )
        if ppl is None:
            continue

        ppls.append(ppl)
        results.append({"id": i, "title": example.get("title", ""), "ppl": ppl})

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ds)}] mean_ppl={np.mean(ppls):.2f}")

    mean_ppl = float(np.mean(ppls))
    std_ppl = float(np.std(ppls))
    median_ppl = float(np.median(ppls))

    print(f"\nPerplexity: {mean_ppl:.2f} ± {std_ppl:.2f}  (median {median_ppl:.2f})")
    print(f"n_articles: {len(ppls)}")

    output = {
        "model": args.model_name,
        "adapter": args.adapter_path,
        "total_ut_steps": args.total_ut_steps,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "n_articles": len(ppls),
        "mean_ppl": mean_ppl,
        "std_ppl": std_ppl,
        "median_ppl": median_ppl,
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()
