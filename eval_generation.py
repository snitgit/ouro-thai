"""eval_generation.py — Thai instruction-following generation for Ouro ut_steps ablation.

Generates responses to a fixed set of Thai instruction prompts from each model variant.
Outputs are saved to JSON for qualitative analysis and LLM-as-judge scoring (eval_judge.py).

Usage:
    # Fine-tuned model
    python eval_generation.py \\
        --adapter_path output/exp2-wangchan-ut1/final_adapter \\
        --total_ut_steps 1 \\
        --output_file results/gen_ft_ut1.json

    # Base model (no adapter)
    python eval_generation.py \\
        --total_ut_steps 4 \\
        --output_file results/gen_base.json
"""

import argparse
import json
import os
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_response(model, tokenizer, instruction, max_new_tokens, device):
    """Generate a response for a single instruction using greedy decoding."""
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,  # required for Ouro UniversalTransformerCache
        )

    new_tokens = output_ids[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


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
        "--prompts_file",
        type=str,
        default="prompts/thai_instruct_bench.json",
        help="JSON file containing instruction prompts.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens to generate per response.",
    )
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    label = f"ft_ut{args.total_ut_steps}" if args.adapter_path else "base"
    print(f"=== Ouro Generation Eval: {label} ===")
    print(f"Model:          {args.model_name}")
    print(f"Adapter:        {args.adapter_path or '(none — base model)'}")
    print(f"ut_steps:       {args.total_ut_steps}")
    print(f"Prompts file:   {args.prompts_file}")
    print(f"max_new_tokens: {args.max_new_tokens}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer from adapter dir (has chat template) or base model
    tok_source = args.adapter_path if args.adapter_path else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (full bf16 — no quantization for eval)
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

    # Load prompts
    with open(args.prompts_file, encoding="utf-8") as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts\n")

    results = []
    for i, item in enumerate(prompts):
        t0 = time.time()
        response = generate_response(
            model, tokenizer, item["instruction"], args.max_new_tokens, device
        )
        elapsed = time.time() - t0

        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "instruction": item["instruction"],
                "response": response,
                "elapsed_sec": round(elapsed, 1),
            }
        )

        preview = response[:120].replace("\n", " ")
        print(f"  [{i+1}/{len(prompts)}] {item['id']} ({elapsed:.1f}s)")
        print(f"    {preview}...")

    output = {
        "model": args.model_name,
        "adapter": args.adapter_path,
        "total_ut_steps": args.total_ut_steps,
        "label": label,
        "max_new_tokens": args.max_new_tokens,
        "n_prompts": len(results),
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {args.output_file}")


if __name__ == "__main__":
    main()
