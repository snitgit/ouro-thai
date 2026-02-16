import argparse
import os
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from trl.trainer.utils import DataCollatorForChatML
from data_config import load_dataset_for_chatml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=os.environ.get("CONFIG", "configs/default.yaml"))
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_dir", type=str,
                        default=os.environ.get("OUTPUT_DIR"))
    parser.add_argument("--total_ut_steps", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--report_to", type=str,
                        default=os.environ.get("REPORT_TO"))
    return parser.parse_args()


def load_config(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    for key in ["model_name", "dataset", "output_dir", "total_ut_steps",
                "max_steps", "num_train_epochs", "report_to"]:
        val = getattr(args, key)
        if val is not None:
            config[key] = val

    return config


def main():
    args = parse_args()
    config = load_config(args)

    model_name = config["model_name"]
    total_ut_steps = config.get("total_ut_steps", 4)
    output_dir = config.get("output_dir", "./output")

    print(f"Model: {model_name}")
    print(f"total_ut_steps: {total_ut_steps}")
    print(f"Output: {output_dir}")

    use_qlora = config.get("use_qlora", True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with configurable loop steps
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            total_ut_steps=total_ut_steps,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            total_ut_steps=total_ut_steps,
            trust_remote_code=True,
        )

    # LoRA config
    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset (returns raw messages for DataCollatorForChatML)
    dataset = load_dataset_for_chatml(
        config.get("dataset", "airesearch/WangchanThaiInstruct"),
    )

    # Training arguments (SFTConfig extends TrainingArguments with dataset_kwargs)
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.get("num_train_epochs", 3),
        max_steps=config.get("max_steps", -1),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 8),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        lr_scheduler_type=config.get("lr_scheduler", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.05),
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 200),
        save_total_limit=3,
        report_to=config.get("report_to", "wandb"),
        optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
        seed=42,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    # Data collator: only compute loss on assistant responses
    collator = DataCollatorForChatML(
        tokenizer=tokenizer,
        max_length=config.get("max_seq_length", 2048),
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    # Save LoRA adapter
    trainer.model.save_pretrained(f"{output_dir}/final_adapter")
    tokenizer.save_pretrained(f"{output_dir}/final_adapter")
    print(f"Adapter saved to {output_dir}/final_adapter")


if __name__ == "__main__":
    main()
