from datasets import load_dataset


DATASET_CONFIGS = {
    "airesearch/WangchanThaiInstruct": {
        "format": "wangchan",
    },
    "pythainlp/thai-aligninstruct-dataset": {
        "format": "alpaca",
    },
}


def messages_wangchan(example):
    """Extract messages from WangchanThaiInstruct dataset.

    Note: WangchanThaiInstruct uses capitalized field names (Instruction, Input, Output).
    """
    instruction = example.get("Instruction", "") or ""
    inp = example.get("Input", "") or ""
    output = example.get("Output", "") or ""
    if inp:
        instruction = f"{instruction}\n\n{inp}"
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]
    return {"messages": messages}


def messages_alpaca(example):
    """Extract messages from Alpaca-style dataset."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    if inp:
        instruction = f"{instruction}\n\n{inp}"
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]
    return {"messages": messages}


FORMATTERS = {
    "wangchan": messages_wangchan,
    "alpaca": messages_alpaca,
}


def load_dataset_for_chatml(dataset_name):
    """Load dataset and return raw messages for DataCollatorForChatML."""
    ds = load_dataset(dataset_name, split="train")

    config = DATASET_CONFIGS.get(dataset_name, {"format": "alpaca"})
    fmt = config["format"]
    formatter = FORMATTERS[fmt]

    ds = ds.map(
        formatter,
        remove_columns=[c for c in ds.column_names if c != "messages"],
        load_from_cache_file=False,
    )

    return ds
