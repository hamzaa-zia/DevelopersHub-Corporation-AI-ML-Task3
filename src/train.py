"""Fine-tune Mistral 7B with QLoRA on the small prepared dataset.

This script trains only LoRA adapter weights, not the full Mistral 7B model.
QLoRA keeps the base model in 4-bit precision to reduce VRAM usage, then trains
small adapter layers on top of it.

Use this script with `data/train_clean_v3.jsonl`. The output folder is
`outputs/mistral-mental-health-lora-safe-v3`, so v3 training does not overwrite
the older safe-v2 adapter.
"""

from __future__ import annotations

import json
import sys

from config import (
    DATA_PATH,
    LORA_OUTPUT_DIR,
    MAX_SEQUENCE_LENGTH,
    MODEL_NAME,
)
from utils import create_folder, print_section


BANNED_DATASET_PHRASES = [
    "is worth taking seriously",
    "makes sense",
    "a response like that deserves care",
    "deserves care instead of pressure",
    "give that reaction some room",
    "let yourself acknowledge it while you decide what comes next",
    "you can take the situation seriously",
]


def check_training_file() -> None:
    """Stop early if the small training dataset has not been created yet."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training file not found: {DATA_PATH}\n"
            "Run `python src/prepare_dataset.py`, then "
            "`python src/clean_dataset.py` before training."
        )


def check_dataset_is_clean() -> None:
    """Stop training if the selected JSONL still has old robotic labels."""
    bad_rows: list[tuple[int, str]] = []

    with DATA_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            assistant_response = row.get("assistant_response", "").lower()

            if any(phrase in assistant_response for phrase in BANNED_DATASET_PHRASES):
                bad_rows.append((line_number, row.get("assistant_response", "")))

    if bad_rows:
        examples = "\n".join(
            f"- Line {line_number}: {response}"
            for line_number, response in bad_rows[:5]
        )
        raise ValueError(
            "The selected training dataset still contains old robotic labels.\n"
            f"Dataset: {DATA_PATH}\n"
            f"Bad rows found: {len(bad_rows)}\n"
            f"{examples}\n\n"
            "Do not train yet. Replace data/train_clean_v3.jsonl with the cleaned "
            "version or run `python src/clean_dataset.py` first."
        )


def check_gpu_available() -> None:
    """Warn clearly when local hardware is not suitable for Mistral 7B QLoRA."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "The 'torch' library is not installed. Run "
            "`pip install -r requirements.txt` first. For actual Mistral 7B "
            "training, install the CUDA-compatible PyTorch build in Colab or "
            "on an NVIDIA GPU machine."
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was not detected. Mistral 7B QLoRA training is too heavy "
            "for normal CPU training.\n\n"
            "Use Google Colab with a GPU runtime, or run this on a machine with "
            "an NVIDIA GPU and enough VRAM."
        )

    gpu_name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {gpu_name}")


def load_small_dataset():
    """Load JSONL data and split it into train/validation sets."""
    from datasets import load_dataset

    print_section("Loading Dataset")
    print(f"Dataset path: {DATA_PATH}")

    dataset = load_dataset(
        "json",
        data_files=str(DATA_PATH),
        split="train",
    )

    # A validation set lets us check whether training loss also improves on
    # examples that the model is not directly training on.
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"Training rows: {len(split_dataset['train'])}")
    print(f"Validation rows: {len(split_dataset['test'])}")
    return split_dataset


def load_tokenizer():
    """Load the tokenizer that turns text into model tokens."""
    from transformers import AutoTokenizer

    print_section("Loading Tokenizer")
    print(f"Tokenizer model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Mistral does not always define a padding token. Using EOS for padding is
    # common for decoder-only language models during fine-tuning.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer


def load_quantized_model():
    """Load Mistral 7B in 4-bit mode to reduce VRAM usage."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    print_section("Loading 4-bit Quantized Model")
    print("Using bitsandbytes 4-bit NF4 quantization for limited VRAM.")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )

    # Disable cache during training to reduce memory use and avoid warnings.
    model.config.use_cache = False

    return model


def build_lora_config():
    """Create LoRA settings for parameter-efficient fine-tuning."""
    from peft import LoraConfig

    print_section("Configuring LoRA")
    print("Only small adapter layers will be trained.")

    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def build_training_arguments():
    """Create memory-conscious training settings for a small test run."""
    from transformers import TrainingArguments

    print_section("Preparing Training Arguments")
    print(f"Adapter output folder: {LORA_OUTPUT_DIR}")

    return TrainingArguments(
        output_dir=str(LORA_OUTPUT_DIR),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        max_steps=100,
        logging_steps=10,
        eval_steps=25,
        save_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        # Disable Trainer mixed precision completely. The 4-bit quantization
        # compute dtype above still stays torch.float16.
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        gradient_checkpointing=True,
    )


def build_trainer(model, tokenizer, dataset, lora_config, training_args):
    """Build TRL's SFTTrainer for supervised fine-tuning."""
    from trl import SFTTrainer

    print_section("Building SFTTrainer")
    print("SFTTrainer reads the `text` field and trains the model to continue it.")

    try:
        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            peft_config=lora_config,
            dataset_text_field="text",
            max_seq_length=MAX_SEQUENCE_LENGTH,
            packing=False,
        )
    except TypeError:
        # Newer TRL versions moved tokenizer/max_seq_length into processing args.
        return SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            peft_config=lora_config,
        )


def main() -> None:
    """Run the full small QLoRA fine-tuning workflow."""
    try:
        print_section("Starting Mistral 7B QLoRA Fine-tuning")
        print(f"Dataset: {DATA_PATH}")
        print(f"Output: {LORA_OUTPUT_DIR}")
        print("Full dataset training is intentionally not used yet.")

        check_training_file()
        check_dataset_is_clean()
        check_gpu_available()
        create_folder(LORA_OUTPUT_DIR)

        dataset = load_small_dataset()
        tokenizer = load_tokenizer()
        model = load_quantized_model()
        lora_config = build_lora_config()
        training_args = build_training_arguments()
        trainer = build_trainer(model, tokenizer, dataset, lora_config, training_args)

        print_section("Training")
        trainer.train()

        print_section("Saving LoRA Adapter")
        trainer.model.save_pretrained(LORA_OUTPUT_DIR)
        tokenizer.save_pretrained(LORA_OUTPUT_DIR)
        print(f"Saved adapters to: {LORA_OUTPUT_DIR}")

    except Exception as error:
        print("\nTraining did not start or did not finish.")
        print(f"Reason: {error}")
        print("\nFor Mistral 7B QLoRA, Colab GPU is recommended if local CUDA is unavailable.")
        sys.exit(1)


if __name__ == "__main__":
    main()
