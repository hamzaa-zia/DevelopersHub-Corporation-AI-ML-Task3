# Mental Health Support Chatbot

Fine-tuning workflow for an emotional support chatbot using Mistral 7B, QLoRA, Hugging Face Transformers, PEFT, TRL, bitsandbytes, and EmpatheticDialogues.

This chatbot is for emotional support only. It does not diagnose, treat, or replace professional mental health care. Self-harm or immediate danger prompts are handled by a fixed safety response before model generation.

## Project Status

The project currently supports:

- Preparing a 3000-row small training dataset from EmpatheticDialogues.
- Cleaning dataset artifacts and weak continuation-style labels.
- Generating assistant-style supportive responses for supervised fine-tuning.
- Training a QLoRA adapter with TRL `SFTTrainer`.
- Loading the fine-tuned LoRA adapter for inference.
- Running safety checks before generation.
- Evaluating responses for safety, repetition, and contextual relevance.

Current active adapter:

```text
outputs/mistral-mental-health-lora-safe-v3
```

The adapter is kept locally and ignored by Git. Later, it can be uploaded to Hugging Face Hub and loaded from there instead of storing it in the repository.

## Project Structure

```text
mental-health-chatbot/
|-- data/
|   |-- train_small.jsonl
|   `-- train_clean_v3.jsonl
|-- outputs/
|   `-- mistral-mental-health-lora-safe-v3/   # local only, ignored by Git
|-- src/
|   |-- __init__.py
|   |-- clean_dataset.py
|   |-- config.py
|   |-- evaluate_model.py
|   |-- evaluate_responses.py
|   |-- inference.py
|   |-- inspect_dataset.py
|   |-- prepare_dataset.py
|   |-- safety.py
|   |-- train.py
|   |-- utils.py
|   `-- validate_dataset_quality.py
|-- .gitignore
|-- AGENTS.md                              # local only, ignored by Git
|-- README.md
`-- requirements.txt
```

## Major Files

- `src/config.py`: Central settings for project paths, base model, adapter path, Hugging Face cache, and small-run training limits.
- `src/prepare_dataset.py`: Loads EmpatheticDialogues, cleans text artifacts, filters weak rows, creates supportive assistant labels, formats examples in Mistral chat style, and saves `data/train_small.jsonl`.
- `src/clean_dataset.py`: Local-only v3 cleanup script. It reads `data/train_small.jsonl`, rewrites repeated/template assistant responses, and saves `data/train_clean_v3.jsonl` without loading Hugging Face datasets.
- `src/inspect_dataset.py`: Prints row count, average text length, shortest and longest examples, random samples, and Mistral format consistency.
- `src/validate_dataset_quality.py`: Checks for short responses, robotic phrases, continuation responses, own storytelling, therapeutic template wording, and invalid Mistral format.
- `src/evaluate_responses.py`: Prints sampled dataset labels with lightweight quality comments before training.
- `src/train.py`: Runs QLoRA fine-tuning on `data/train_clean_v3.jsonl` and saves LoRA adapter files to `outputs/mistral-mental-health-lora-safe-v3`.
- `src/inference.py`: Loads the base Mistral model from cache, attaches the LoRA adapter, checks self-harm safety first, and generates responses for test prompts.
- `src/evaluate_model.py`: Tests the fine-tuned chatbot on 20 prompts and reports safety, repetition, and contextual checks.
- `src/safety.py`: Stores crisis keywords and the fixed self-harm safety response.
- `src/utils.py`: Small shared helper functions.

## Libraries Used

- `torch`: Deep learning framework used by Transformers for model execution.
- `transformers`: Loads Mistral 7B, tokenizers, quantization config, and training arguments.
- `datasets`: Loads EmpatheticDialogues and JSONL training data.
- `accelerate`: Helps Transformers place model parts across available hardware.
- `peft`: Adds and loads LoRA adapters for parameter-efficient fine-tuning.
- `trl`: Provides `SFTTrainer` for supervised fine-tuning.
- `bitsandbytes`: Enables 4-bit quantization for QLoRA.
- `pandas`: Cleans and filters dataset rows.
- `scikit-learn`: Available for lightweight evaluation or future splitting utilities.
- `streamlit`: Reserved for a future local chatbot interface.

## Setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Mistral 7B QLoRA requires a CUDA-capable NVIDIA GPU. If local CUDA is not available, use a GPU runtime such as Google Colab for training.

## Workflow

Prepare the small dataset:

```bash
python src/prepare_dataset.py
```

Inspect the generated JSONL:

```bash
python src/inspect_dataset.py
```

Create the local-only v3 cleaned dataset:

```bash
python src/clean_dataset.py
```

Run the dataset quality gate:

```bash
python src/validate_dataset_quality.py
```

Review sampled generated labels:

```bash
python src/evaluate_responses.py
```

Train the QLoRA adapter:

```bash
python src/train.py
```

Run inference after the adapter exists:

```bash
python src/inference.py
```

Evaluate the fine-tuned chatbot:

```bash
python src/evaluate_model.py
```

## Configuration

Main settings live in `src/config.py`.

Defaults:

```text
BASE_MODEL=mistralai/Mistral-7B-v0.1
ADAPTER_PATH=outputs/mistral-mental-health-lora-safe-v3
DATA_PATH=data/train_clean_v3.jsonl
HF_CACHE_DIR=/root/.cache/huggingface
HF_HUB_CACHE=/root/.cache/huggingface/hub
HF_LOCAL_FILES_ONLY=false
```

You can override these with environment variables when needed. For example:

```bash
set ADAPTER_PATH=outputs/mistral-mental-health-lora-safe-v3
set HF_LOCAL_FILES_ONLY=false
```

Set `HF_LOCAL_FILES_ONLY=true` only when you want inference to use already-cached base model files and avoid downloading model weights.

## GitHub Notes

The repository is prepared so source code and documentation can be committed safely, while large local artifacts stay out of Git.

Do not commit:

- Hugging Face base model cache.
- `outputs/` adapter folders unless you explicitly choose to publish them.
- `__pycache__/` or `.pyc` files.
- Temporary archives or generated images.
- `.env` files or secrets.
- `AGENTS.md`.

If you later want to share the LoRA adapter, the cleaner path is to upload it to Hugging Face Hub and mention the adapter repository in this README.

## Safety Rules

- The chatbot provides emotional support only.
- It must not diagnose mental illness.
- It must not claim to treat mental health conditions.
- It must not replace therapy, counseling, emergency care, or professional medical advice.
- It should respond with validation, reassurance, and specific support.
- If self-harm or immediate danger language is detected, it must return the fixed crisis-safe response and skip model generation.
