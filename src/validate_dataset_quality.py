"""Validate prepared dataset quality before fine-tuning.

This script checks that generated assistant labels are aligned with the support
chatbot goal. It verifies that short responses, continuation-style replies,
first-person storytelling, robotic phrases, and format errors are not present.
"""

import json
import re

from config import get_training_data_path
from inspect_dataset import is_mistral_chat_format
from utils import print_section


MIN_ASSISTANT_WORDS = 8

ROBOTIC_PHRASES = [
    "from what you shared",
    "you shared about",
    "as an ai",
    "i conclude",
]

THERAPEUTIC_PHRASES = [
    "minimize that reaction",
    "emotional pressure",
    "grounding",
    "your body reacted",
    "sit with",
    "what happened is worth paying attention to",
    "start with the part that feels hardest right now",
    "start with the part that feels hardest",
    "be a little patient with yourself here",
    "be a little patient with yourself",
    "it makes sense that",
    "that sounds",
    "that is a lot to take in",
    "from what you shared",
    "i can see why this stayed with you",
    "what part of it do you want to hold onto most",
    "what part feels most important to hold onto",
    "try giving yourself a pause before reacting to it",
    "try pausing before",
    "take one slow breath",
    "one steady breath",
    "give yourself room to respond at your own pace",
    "give yourself a little space",
    "handle the next piece slowly",
    "focus on the next small thing you can control",
    "take the pressure off having the perfect reaction",
    "is worth taking seriously",
    "makes sense",
    "a response like that deserves care",
    "deserves care instead of pressure",
    "you can take the situation seriously",
    "you can take it seriously",
    "give that reaction some room",
    "let yourself acknowledge it while you decide what comes next",
    "can feel genuinely",
    "can feel terrifying",
    "can feel meaningful",
    "can feel painful",
    "can catch you off guard",
    "can make it hard to feel",
    "thinking about your",
    "dealing with your parents",
    "dealing with your mother",
    "dealing with your father",
    "dealing with your friend",
    "dealing with your girlfriend",
    "dealing with your boyfriend",
    "dealing with your husband",
    "dealing with your wife",
    "dealing with the birthday plans",
    "dealing with the job news",
    "dealing with work",
    "your parents surprising you with a new car can",
    "your neighbor's house is haunted, you have seen ghosts can",
    "what happened",
    "the part about what happened",
    "the part about",
    "this stayed with you",
]

CONTINUATION_PHRASES = [
    "that happened to me",
    "i had the same",
    "i remember when i",
    "one time i",
    "when i was",
    "my friend did",
    "my mom did",
    "my dad did",
]

OWN_STORY_PATTERNS = [
    r"\bi remember\b",
    r"\bi was\b",
    r"\bi had\b",
    r"\bmy (mother|father|mom|dad|friend|girlfriend|boyfriend|husband|wife)\b",
]


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace for validation checks."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def normalize_for_duplicate_check(text: str) -> str:
    """Normalize text so duplicate user prompts can be detected."""
    text = normalize(text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_rows() -> list[dict[str, str]]:
    """Load prepared training rows from JSONL."""
    rows: list[dict[str, str]] = []
    dataset_path = get_training_data_path()
    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            row = json.loads(line)
            row["line_number"] = line_number
            rows.append(row)
    return rows


def contains_any(text: str, phrases: list[str]) -> bool:
    """Return True when any phrase appears in the text."""
    text = normalize(text)
    return any(phrase in text for phrase in phrases)


def has_own_storytelling(text: str) -> bool:
    """Check whether the assistant appears to tell its own personal story."""
    text = normalize(text)
    return any(re.search(pattern, text) for pattern in OWN_STORY_PATTERNS)


def collect_failures(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Collect rows that fail each quality rule."""
    failures: dict[str, list[dict[str, str]]] = {
        "short_assistant_response": [],
        "robotic_phrase": [],
        "continuation_response": [],
        "own_storytelling": [],
        "overly_therapeutic_phrase": [],
        "invalid_mistral_format": [],
        "duplicate_user_message": [],
        "duplicate_assistant_response": [],
    }
    seen_user_messages: set[str] = set()
    seen_assistant_responses: set[str] = set()

    for row in rows:
        assistant_response = row.get("assistant_response", "")
        text = row.get("text", "")
        user_message = row.get("user_message", "")
        duplicate_key = normalize_for_duplicate_check(user_message)
        response_key = normalize_for_duplicate_check(assistant_response)

        if len(assistant_response.split()) < MIN_ASSISTANT_WORDS:
            failures["short_assistant_response"].append(row)
        if contains_any(assistant_response, ROBOTIC_PHRASES):
            failures["robotic_phrase"].append(row)
        if contains_any(assistant_response, CONTINUATION_PHRASES):
            failures["continuation_response"].append(row)
        if has_own_storytelling(assistant_response):
            failures["own_storytelling"].append(row)
        if contains_any(assistant_response, THERAPEUTIC_PHRASES):
            failures["overly_therapeutic_phrase"].append(row)
        if not is_mistral_chat_format(text):
            failures["invalid_mistral_format"].append(row)
        if duplicate_key in seen_user_messages:
            failures["duplicate_user_message"].append(row)
        else:
            seen_user_messages.add(duplicate_key)
        if response_key in seen_assistant_responses:
            failures["duplicate_assistant_response"].append(row)
        else:
            seen_assistant_responses.add(response_key)

    return failures


def print_failure_samples(name: str, rows: list[dict[str, str]]) -> None:
    """Print up to three example failures for one rule."""
    if not rows:
        return

    print(f"\n{name} examples:")
    for row in rows[:3]:
        print(f"- Line {row['line_number']}: {row.get('assistant_response', '')}")


def main() -> None:
    """Run dataset quality validation."""
    rows = load_rows()
    failures = collect_failures(rows)

    print_section("Dataset Quality Validation")
    print(f"Dataset path: {get_training_data_path()}")
    print(f"Total rows: {len(rows)}")
    print(f"Minimum assistant response words: {MIN_ASSISTANT_WORDS}")

    for name, failed_rows in failures.items():
        print(f"{name}: {len(failed_rows)}")

    all_clean = all(len(failed_rows) == 0 for failed_rows in failures.values())
    print(f"Quality gate passed: {all_clean}")

    for name, failed_rows in failures.items():
        print_failure_samples(name, failed_rows)


if __name__ == "__main__":
    main()
