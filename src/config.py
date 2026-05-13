"""Central project settings.

This file keeps paths and model names in one place. Beginners should edit this
file first instead of changing the same value in multiple scripts.
"""

import os
from pathlib import Path


# Root folder of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str | None, default_path: Path) -> Path:
    """Return an absolute path from an environment value or project default."""
    path = Path(value).expanduser() if value else default_path
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


# Folder for prepared training files.
DATA_DIR = PROJECT_ROOT / "data"

# Small Hugging Face dataset output used before full training.
TRAIN_SMALL_PATH = DATA_DIR / "train_small.jsonl"

# Locally cleaned v3 dataset used for the next safer fine-tuning pass.
TRAIN_CLEAN_V3_PATH = DATA_DIR / "train_clean_v3.jsonl"

# Explicit training dataset for the v3 fine-tuning pass.
DATA_PATH = TRAIN_CLEAN_V3_PATH

# Folder where trained adapters and logs can be saved later.
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Base model used for QLoRA training and adapter inference.
BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-v0.1")

# Backward-compatible name used by the training script.
MODEL_NAME = BASE_MODEL

# Active LoRA adapter. This stays local and is ignored by Git.
ADAPTER_PATH = resolve_project_path(
    os.getenv("ADAPTER_PATH"),
    OUTPUT_DIR / "mistral-mental-health-lora-safe-v3",
)

# Backward-compatible name used by the training script.
LORA_OUTPUT_DIR = ADAPTER_PATH

# Hugging Face cache settings. Defaults support the Colab/Linux cache path, but
# the values can be changed with environment variables on Windows or another PC.
HF_CACHE_DIR = Path(os.getenv("HF_CACHE_DIR", "/root/.cache/huggingface")).expanduser()
HF_HUB_CACHE = Path(os.getenv("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))).expanduser()
HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "false").lower() == "true"

# Local CSV backups can exist, but the active workflow uses Hugging Face
# EmpatheticDialogues through `src/prepare_dataset.py`.
LEGACY_CSV_PATH = PROJECT_ROOT / "emotion-emotion_69k.csv"

# Small defaults for early testing before full training.
MAX_TEST_ROWS = 100
SMALL_TRAINING_SAMPLE_SIZE = 3000
MAX_SEQUENCE_LENGTH = 512


def get_training_data_path() -> Path:
    """Return the explicit v3 training dataset path."""
    return DATA_PATH
