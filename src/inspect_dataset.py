"""Inspect the prepared JSONL training dataset.

This script checks the small training file before fine-tuning. It prints basic
dataset statistics, shows the shortest and longest examples, displays five
random samples, and verifies that every row follows the Mistral chat format.
"""

import json
import random
import sys

from config import get_training_data_path
from utils import print_section


RANDOM_SEED = 42
SAMPLE_COUNT = 5

BANNED_DATASET_PHRASES = [
    "is worth taking seriously",
    "makes sense",
    "a response like that deserves care",
    "deserves care instead of pressure",
    "give that reaction some room",
    "let yourself acknowledge it while you decide what comes next",
    "you can take the situation seriously",
]


def load_jsonl_dataset() -> list[dict[str, str]]:
    """Load the JSONL file into a list of dictionaries."""
    rows: list[dict[str, str]] = []

    dataset_path = get_training_data_path()
    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            row["line_number"] = line_number
            rows.append(row)

    return rows


def is_mistral_chat_format(text: str) -> bool:
    """Check whether one text field follows the expected Mistral format."""
    required_parts = ["<s>[INST]", "<<SYS>>", "<</SYS>>", "[/INST]", "</s>"]

    if not all(part in text for part in required_parts):
        return False

    return (
        text.startswith("<s>[INST]")
        and text.endswith("</s>")
        and text.index("<s>[INST]") < text.index("<<SYS>>")
        and text.index("<<SYS>>") < text.index("<</SYS>>")
        and text.index("<</SYS>>") < text.index("[/INST]")
        and text.index("[/INST]") < text.index("</s>")
    )


def contains_banned_dataset_phrase(text: str) -> bool:
    """Check for old robotic labels that should never be trained again."""
    lowered_text = text.lower()
    return any(phrase in lowered_text for phrase in BANNED_DATASET_PHRASES)


def find_banned_phrase_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return rows that still contain stale response templates."""
    return [
        row
        for row in rows
        if contains_banned_dataset_phrase(row.get("assistant_response", ""))
    ]


def print_example(title: str, row: dict[str, str]) -> None:
    """Print one dataset example in a readable way."""
    print_section(title)
    print(f"Line number: {row['line_number']}")
    print(f"Text length: {len(row['text'])}")
    print(f"User message: {row.get('user_message', '')}")
    print(f"Assistant response: {row.get('assistant_response', '')}")
    print("\nFormatted text:")
    print(row["text"])


def main() -> None:
    """Run all dataset inspection checks."""
    rows = load_jsonl_dataset()

    if not rows:
        print("No rows found in the dataset.")
        return

    text_lengths = [len(row["text"]) for row in rows]
    shortest_row = min(rows, key=lambda row: len(row["text"]))
    longest_row = max(rows, key=lambda row: len(row["text"]))
    invalid_rows = [
        row for row in rows if not is_mistral_chat_format(row.get("text", ""))
    ]
    banned_phrase_rows = find_banned_phrase_rows(rows)

    print_section("Dataset Summary")
    print(f"Dataset path: {get_training_data_path()}")
    print(f"Total rows: {len(rows)}")
    print(f"Average text length: {sum(text_lengths) / len(text_lengths):.2f}")
    print(f"Mistral format consistent: {len(invalid_rows) == 0}")
    print(f"Invalid format rows: {len(invalid_rows)}")
    print(f"Banned old-template rows: {len(banned_phrase_rows)}")

    if banned_phrase_rows:
        print_section("Banned Old-Template Rows")
        for row in banned_phrase_rows[:SAMPLE_COUNT]:
            print(f"Line number: {row['line_number']}")
            print(f"Assistant response: {row.get('assistant_response', '')}\n")
        print("Stop here. Regenerate or replace data/train_clean_v3.jsonl before training.")
        sys.exit(1)

    print_example("Shortest Example", shortest_row)
    print_example("Longest Example", longest_row)

    random.seed(RANDOM_SEED)
    random_samples = random.sample(rows, k=min(SAMPLE_COUNT, len(rows)))

    print_section("Five Random Samples")
    for index, row in enumerate(random_samples, start=1):
        print(f"\nSample {index}")
        print(f"Line number: {row['line_number']}")
        print(f"Text length: {len(row['text'])}")
        print(f"User message: {row.get('user_message', '')}")
        print(f"Assistant response: {row.get('assistant_response', '')}")
        print(f"Format valid: {is_mistral_chat_format(row.get('text', ''))}")

    if invalid_rows:
        print_section("Invalid Format Rows")
        for row in invalid_rows[:SAMPLE_COUNT]:
            print(f"Line number: {row['line_number']}")


if __name__ == "__main__":
    main()
