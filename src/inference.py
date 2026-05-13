"""Run inference with the fine-tuned Mistral 7B LoRA adapter.

This script loads the base Mistral 7B model in 4-bit mode, attaches the trained
LoRA adapter, formats prompts the same way as training, and prints only the
assistant response for each test prompt.
"""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import ADAPTER_PATH, BASE_MODEL, HF_CACHE_DIR, HF_HUB_CACHE, HF_LOCAL_FILES_ONLY
from safety import get_safe_response

# Set Hugging Face cache variables before loading models or tokenizers.
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)


TEST_PROMPTS = [
    "I feel stressed and overwhelmed.",
    "I feel lonely these days.",
    "I failed my interview and feel useless.",
    "I am angry at my friend.",
    "I feel like hurting myself.",
]


SYSTEM_MESSAGE = (
    "You are an emotional support chatbot. Provide reassurance, emotional "
    "support, and validation. Understand the user's feeling first, then respond "
    "clearly and specifically. Do not diagnose, treat, or replace professional "
    "mental health care. If the user expresses self-harm or immediate danger, "
    "encourage them to contact emergency services or a trusted person immediately."
)


def is_self_harm(text):
    """Check for self-harm language before model generation."""
    return get_safe_response(text) is not None


def build_mistral_prompt(user_text: str) -> str:
    """Create a direct support prompt that discourages repeated templates."""
    prompt = f"""
You are a warm, emotionally supportive chatbot.

Reply in 2 to 3 natural sentences.
Mention the user's specific situation directly.
Do not ask reflective questions.
Do not use repeated phrases like:
- I can see why this stayed with you
- What part of it do you want to hold onto most
- That is a lot to take in
- Try giving yourself a pause before reacting to it
- Take one slow breath

User: {user_text}
Assistant:
"""
    return prompt.strip()


def resolve_adapter_files_path():
    """Find the folder that actually contains the LoRA adapter files.

    Some zip downloads extract into an extra nested folder. This keeps the
    public adapter root stable while still loading the real adapter files.
    """
    candidates = [
        ADAPTER_PATH,
        ADAPTER_PATH / ADAPTER_PATH.name,
    ]

    for candidate in candidates:
        if (candidate / "adapter_config.json").exists():
            return candidate

    raise FileNotFoundError(
        f"LoRA adapter files were not found under: {ADAPTER_PATH}\n"
        "Expected an adapter_config.json file in the adapter folder."
    )


def load_tokenizer():
    """Load tokenizer from the adapter folder when it exists.

    The adapter folder should contain tokenizer files after training. If they
    are missing, the script falls back to the base Mistral tokenizer.
    """
    adapter_files_path = resolve_adapter_files_path()

    if (adapter_files_path / "tokenizer.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(adapter_files_path, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            use_fast=True,
            cache_dir=str(HF_HUB_CACHE),
            local_files_only=HF_LOCAL_FILES_ONLY,
        )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model():
    """Load Mistral 7B with bitsandbytes 4-bit quantization."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        cache_dir=str(HF_HUB_CACHE),
        local_files_only=HF_LOCAL_FILES_ONLY,
    )
    model.config.use_cache = True
    return model


def load_model_with_adapter():
    """Load the base model and attach the trained LoRA adapter."""
    from peft import PeftModel

    adapter_files_path = resolve_adapter_files_path()
    base_model = load_base_model()
    model = PeftModel.from_pretrained(base_model, adapter_files_path)
    model.eval()
    return model


def generate_response(user_prompt: str, model, tokenizer) -> str:
    """Generate one assistant response for a user prompt."""
    safe_response = get_safe_response(user_prompt)
    if safe_response:
        return safe_response

    prompt = build_mistral_prompt(user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.8,
            top_p=0.92,
            repetition_penalty=1.25,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    response_ids = output_ids[0][prompt_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response.strip()


def main() -> None:
    """Load the model once and print only assistant responses."""
    tokenizer = load_tokenizer()
    model = load_model_with_adapter()

    for prompt in TEST_PROMPTS:
        print(f"Prompt: {prompt}")
        print(generate_response(prompt, model, tokenizer))
        print()


if __name__ == "__main__":
    main()
