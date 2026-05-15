# Mental Health Support Chatbot

A mental health support chatbot project that demonstrates dataset preparation,
label engineering, QLoRA fine-tuning workflow, safety handling, response
evaluation, and a CPU-safe Streamlit demo interface.

The chatbot is designed for emotional support only. It does not diagnose,
treat, or replace therapy, counseling, emergency care, or professional medical
advice. Self-harm or immediate danger prompts are handled through explicit
safety triggers before any normal response is returned.

---

## Project Overview

This project fine-tunes a Mistral 7B-based emotional support chatbot using the
[Empathetic Dialogues (Facebook AI) dataset from Kaggle](https://www.kaggle.com/datasets/atharvjairath/empathetic-dialogues-facebook-ai),
Hugging Face Transformers, PEFT, TRL, bitsandbytes, and QLoRA. The work also
includes a safe local demo app so reviewers can run the interface without
needing a GPU.

The full Mistral 7B adapter workflow requires a CUDA GPU or Google Colab GPU
runtime. The local Streamlit app is clearly marked as a **keyword-based demo**
and is provided only to show the chatbot interface, safety behavior, and
reviewer-friendly interaction flow.

---

## Main Submission Modes

| Mode | Purpose | Hardware | Entry Point |
|---|---|---|---|
| Safe Demo Mode | Shows the chatbot UI, safety behavior, and sample support responses | CPU is enough | `streamlit run app.py` |
| Full Adapter Chat UI | Tests the real Mistral 7B LoRA adapter inside Streamlit | CUDA GPU / Colab GPU | `streamlit run app.py`, then select `Full Mistral Adapter (GPU)` |
| Full Model Inference Script | Loads Mistral 7B with the trained LoRA adapter in the terminal | CUDA GPU / Colab GPU | `python src/inference.py` |
| Training Workflow | Rebuilds the fine-tuning process | CUDA GPU / Colab GPU | `python src/train.py` |

---

## Dataset Source and Sampling

The dataset source used for this project was
[Empathetic Dialogues (Facebook AI) on Kaggle](https://www.kaggle.com/datasets/atharvjairath/empathetic-dialogues-facebook-ai).
The workflow started with a smaller sample first, then expanded during later
training attempts:

| Step | Dataset Work |
|---|---|
| Initial sample | Used a smaller 1000-row working sample to test cleaning, formatting, and early fine-tuning behavior. |
| Expanded sample | Prepared a larger 3000-row sample for later training/refinement attempts after applying data, label, and format engineering. |
| Current repository artifact | Includes the cleaned small review dataset files, with `data/train_small.jsonl` containing 1000 rows and `data/train_clean_v3.jsonl` containing 994 rows after cleanup and deduplication. |

The dataset was not used as raw chatbot output directly. It was processed into
assistant-style support examples through:

- **data engineering:** cleaning text artifacts, filtering weak examples, and
  removing duplicates,
- **label engineering:** rewriting responses so they sound like supportive
  assistant replies instead of dialogue continuations,
- **format engineering:** converting each row into the Mistral instruction
  format used for supervised fine-tuning.

---

## Tools and Libraries

| Library / Tool | Function in This Project |
|---|---|
| `streamlit` | Builds the local chatbot interface with chat history, input area, demo alert, and keyword-category explanation. |
| `torch` | Runs deep learning operations for Mistral model loading, training, and generation. |
| `transformers` | Loads the tokenizer, Mistral 7B model, quantization config, generation settings, and training arguments. |
| `datasets` | Loads EmpatheticDialogues and local JSONL training files. |
| `accelerate` | Helps place model components on available GPU hardware during training/inference. |
| `peft` | Adds and loads LoRA adapters for parameter-efficient fine-tuning. |
| `trl` | Provides `SFTTrainer` for supervised fine-tuning on formatted instruction-response data. |
| `bitsandbytes` | Enables 4-bit quantization for QLoRA so Mistral 7B can be trained with reduced VRAM. |
| `pandas` | Supports dataset cleaning, filtering, deduplication, and label preparation. |
| Google Colab | Used for GPU-based training because local CPU execution is not practical for Mistral 7B. |
| ChatGPT | Used as a learning aid for understanding concepts, refining approaches, and reasoning through data/model improvement steps. |

---

## Project Structure

```text
mental-health-chatbot/
|-- app.py                         # Streamlit CPU-safe chatbot demo
|-- .streamlit/
|   `-- config.toml                # Streamlit review-friendly defaults
|-- data/
|   |-- train_small.jsonl           # prepared small dataset
|   `-- train_clean_v3.jsonl        # cleaned v3 training dataset
|-- notebooks/
|   `-- submission_demo.ipynb       # lightweight review/Colab guide
|-- src/
|   |-- clean_dataset.py            # local v3 dataset cleanup
|   |-- config.py                   # paths, model names, and cache settings
|   |-- demo_chatbot.py             # CPU-safe keyword demo response engine
|   |-- evaluate_model.py           # prompt-based model evaluation script
|   |-- evaluate_responses.py       # dataset response quality review
|   |-- inference.py                # full Mistral + LoRA inference
|   |-- inspect_dataset.py          # dataset inspection checks
|   |-- prepare_dataset.py          # EmpatheticDialogues preparation
|   |-- safety.py                   # crisis keyword checks and safe response
|   |-- train.py                    # QLoRA fine-tuning script
|   |-- utils.py                    # shared helper functions
|   `-- validate_dataset_quality.py # dataset quality gate
|-- .gitignore
|-- README.md
`-- requirements.txt
```

Local adapter files are stored under `outputs/` and are ignored by Git because
they can be large. If included in final submission, the adapter should be
uploaded separately to Hugging Face Hub.

Uploaded adapter repositories:

- [mistral-mental-health-lora-safe-v2](https://huggingface.co/hamzaa-zia/mistral-mental-health-lora-safe-v2)
- [mistral-mental-health-lora-safe-v3](https://huggingface.co/hamzaa-zia/mistral-mental-health-lora-safe-v3)

---

## End-to-End Workflow

| Stage | What Was Done | Main Files |
|---|---|---|
| 1. Dataset selection | Used the Kaggle Empathetic Dialogues dataset as the emotional conversation source. | `src/prepare_dataset.py` |
| 2. Data cleaning | Replaced dataset artifacts, normalized text, filtered low-context examples, and removed weak rows. | `src/prepare_dataset.py` |
| 3. Label engineering | Built supportive assistant responses instead of directly using continuation-style dialogue labels. | `src/prepare_dataset.py` |
| 4. Format engineering | Converted rows into Mistral instruction format using system message, user message, and assistant response. | `src/prepare_dataset.py` |
| 5. Quality checks | Checked short responses, repeated phrases, invalid format, duplicate prompts, and own-storytelling patterns. | `src/inspect_dataset.py`, `src/validate_dataset_quality.py` |
| 6. Training | Fine-tuned Mistral 7B through QLoRA and LoRA adapter training. | `src/train.py` |
| 7. Iteration | Repeated training/refinement across v1, v2, and v3 to reduce robotic and repeated responses. | `src/clean_dataset.py`, `src/train.py` |
| 8. Safety | Added crisis keyword detection and fixed safe responses for self-harm or immediate-danger prompts. | `src/safety.py`, `src/inference.py` |
| 9. Evaluation | Tested prompts for contextual relevance, repetition, and safety behavior. | `src/evaluate_model.py` |
| 10. Demo interface | Added a Streamlit chatbot UI that runs without GPU for reviewers. | `app.py`, `src/demo_chatbot.py` |

---

## Iteration History: v1 to v3

This project was not completed in one pass. The workflow went through repeated
training and refinement cycles, including Colab runs across **two different
Google accounts** due to GPU/runtime limits. The model and data pipeline were
retrained around **7 to 8 times** while improving response quality and safety.

| Version | Focus | Main Improvement |
|---|---|---|
| v1 | Initial fine-tuning workflow | Built the first dataset preparation, Mistral formatting, and QLoRA training path. |
| v2 | Safer response behavior | Improved support style, reduced weak continuation-style replies, and started stricter safety handling. |
| v3 | Cleaner final dataset | Rewrote repeated/template responses, banned stale phrases, removed duplicate rows, and validated final training data. |

Important refinements:

- Reworked dataset labels so the assistant responds supportively instead of
  sounding like another speaker continuing the dialogue.
- Added banned phrase checks for robotic or repeated wording.
- Rebuilt `train_clean_v3.jsonl` after removing duplicate prompts and duplicate
  assistant responses.
- Added safety-first inference so crisis prompts skip normal generation.
- Used small training runs before heavier training to reduce wasted GPU time.
- Evaluated responses repeatedly for repetition, context, and safety.

---

## Safety Design

The project uses explicit safety handling before model generation.

- `src/safety.py` stores crisis keywords and the fixed crisis response.
- `src/inference.py` checks safety before sending the prompt to the model.
- `src/demo_chatbot.py` also checks the same safety function before returning a
  keyword-based demo response.
- Self-harm or immediate-danger messages do not receive a normal chatbot reply.

Current safety limits:

- The chatbot provides emotional support only.
- It does not diagnose mental illness.
- It does not claim to treat or cure mental health conditions.
- It does not replace therapy, counseling, emergency care, or professional
  medical advice.
- It encourages emergency services or a trusted person for crisis situations.

---

## Streamlit Demo App

Run the Streamlit interface:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app has two selectable modes in the sidebar:

| Streamlit Mode | What It Does | When to Use |
|---|---|---|
| `Keyword Demo (CPU)` | Uses keyword rules from `src/demo_chatbot.py`; does not load Mistral 7B. | Normal laptop review and interface testing. |
| `Full Mistral Adapter (GPU)` | Loads `src/inference.py`, attaches the LoRA adapter, and generates real model responses in the chat UI. | Colab GPU or CUDA GPU testing. |

The demo mode includes:

- one prompt area for users to share feelings or situations,
- conversation-style chat bubbles,
- a visible alert explaining that local demo responses are keyword-based,
- an expandable list of keyword categories used by the demo engine,
- crisis-safety handling for self-harm prompts,
- a clear note that the real Mistral 7B adapter is not loaded in local demo
  mode.

Demo keyword categories include:

- accident / near-death experience,
- academic setback,
- positive friendship,
- loneliness,
- anxiety,
- anger,
- guilt,
- work pressure,
- family,
- health.

This app is intentionally included so reviewers can verify the interface and
safety behavior without GPU access. It is not presented as the real fine-tuned
adapter output.

---

## Real Adapter Testing in Streamlit

To test the real fine-tuned adapter through the chatbot interface:

1. Run the app on Colab GPU or a CUDA-capable machine.
2. Make sure the adapter files are available locally under:

```text
outputs/mistral-mental-health-lora-safe-v3
```

3. Start Streamlit:

```bash
streamlit run app.py
```

4. In the sidebar, select:

```text
Full Mistral Adapter (GPU)
```

5. Send a message in the chat input.

The first response can take time because the app loads Mistral 7B and attaches
the LoRA adapter. Crisis/self-harm prompts are still checked before model
generation.

If the adapter is not stored locally, download it from Hugging Face first or
update `ADAPTER_PATH` in `src/config.py`.

---

## Full Mistral Inference

Full inference loads the base Mistral 7B model and the trained LoRA adapter:

```bash
python src/inference.py
```

Default local adapter path:

```text
outputs/mistral-mental-health-lora-safe-v3
```

Uploaded Hugging Face adapter versions:

| Version | Hugging Face Repository | Notes |
|---|---|---|
| Safe v2 | [hamzaa-zia/mistral-mental-health-lora-safe-v2](https://huggingface.co/hamzaa-zia/mistral-mental-health-lora-safe-v2) | Intermediate safety-focused LoRA adapter version. |
| Safe v3 | [hamzaa-zia/mistral-mental-health-lora-safe-v3](https://huggingface.co/hamzaa-zia/mistral-mental-health-lora-safe-v3) | Cleaner final LoRA adapter version after additional response cleanup and validation. |

This mode requires a CUDA GPU or Colab GPU runtime. Normal CPU-only machines are
not suitable for running Mistral 7B inference.

---

## Training Commands

Run these steps in order:

```bash
python src/prepare_dataset.py
python src/inspect_dataset.py
python src/clean_dataset.py
python src/validate_dataset_quality.py
python src/evaluate_responses.py
python src/train.py
python src/evaluate_model.py
```

The training script is intentionally configured for a small controlled run
before scaling. This helps verify the workflow before spending more GPU time.

---

## What I Learned

Through this project, the work covered practical concepts behind:

- dataset cleaning and filtering,
- label engineering for safer assistant-style responses,
- data engineering for JSONL training files,
- format engineering for Mistral instruction-response examples,
- QLoRA and LoRA adapter-based fine-tuning,
- repeated model refinement after observing weak outputs,
- safety-first chatbot response design,
- separating a reviewer-friendly demo app from GPU-only model inference.

---

## Cleanup Notes

The repository was cleaned so the GitHub submission focuses on useful source
files, documentation, and reproducible workflow files.

Removed because they no longer had a project use:

- `excalidraw.log`: leftover log file unrelated to the chatbot workflow.
- `streamlit_app.log`: generated local Streamlit test log.
- `streamlit_app.err.log`: generated local Streamlit error log.
- `scikit-learn` from `requirements.txt`: it was not imported or used by the
  current codebase.

Kept out of Git intentionally:

- `outputs/`: local LoRA adapter and training artifacts, better uploaded to
  Hugging Face Hub if needed.
- `AGENTS.md`: local development instructions, not part of the public
  submission.

---

## Submission Checklist

For GitHub:

- Include source code, README, requirements, notebook, and small dataset files.
- Do not commit `outputs/`, Hugging Face cache folders, `.env` files, logs, or
  local temporary files.
- Keep the Streamlit app as the CPU-safe reviewer demo.

For Hugging Face:

- Upload the generated LoRA adapter folder if adapter files are part of the
  final submission.
- Share the Hugging Face adapter link along with the GitHub repository link.
- Do not upload private tokens, local cache folders, or unrelated files.

Suggested final submission format:

```text
Hugging Face Adapter v2: https://huggingface.co/hamzaa-zia/mistral-mental-health-lora-safe-v2
Hugging Face Adapter v3: https://huggingface.co/hamzaa-zia/mistral-mental-health-lora-safe-v3
Demo Command: streamlit run app.py
Full Model Note: Mistral 7B inference requires Colab GPU or CUDA GPU.
```
