"""Evaluate regenerated assistant responses.

This script prints 10 random assistant responses and adds simple quality
comments for each one. It is a lightweight review step before model training.
"""

import json
import random
import re

from config import get_training_data_path
from utils import print_section


SAMPLE_COUNT = 10
RANDOM_SEED = 7

VALIDATION_MARKERS = [
    "makes sense",
    "understandable",
    "it is okay",
    "it sounds",
    "that sounds",
    "feeling",
    "i get why",
    "i can see why",
    "i can understand why",
    "sounds like",
    "sounds painful",
    "sounds scary",
    "can feel",
]

GENERIC_POSITIVITY_MARKERS = [
    "just stay positive",
    "everything happens for a reason",
    "look on the bright side",
    "cheer up",
    "don't worry",
]

REPETITIVE_TEMPLATE_MARKERS = [
    "from what you shared about",
    "let yourself take it in",
    "you do not have to minimize",
    "if it feels too heavy to carry alone",
]


def load_rows() -> list[dict[str, str]]:
    """Load prepared JSONL rows."""
    rows: list[dict[str, str]] = []
    dataset_path = get_training_data_path()
    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            row = json.loads(line)
            row["line_number"] = line_number
            rows.append(row)
    return rows


def normalize(text: str) -> str:
    """Lowercase text and collapse whitespace for easier checks."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def has_emotional_validation(response: str) -> bool:
    """Check whether the response includes validation language."""
    response = normalize(response)
    return any(marker in response for marker in VALIDATION_MARKERS)


def is_contextual(user_message: str, response: str) -> bool:
    """Check whether the response adapts to the situation without parroting."""
    user_text = normalize(user_message)
    response_text = normalize(response)

    context_markers = {
        "vacation": ["vacation", "trip", "travel"],
        "health": ["sick", "hospital", "medicine", "vomiting", "choked", "health", "head", "hurt", "injured", "hit"],
        "work": ["job", "jobs", "work", "working", "interview", "boss", "coworker"],
        "school": ["school", "exam", "class", "student", "college"],
        "relationship": ["girlfriend", "boyfriend", "husband", "wife", "dating", "cheated", "marriage"],
        "family": ["mom", "mother", "dad", "father", "daughter", "son", "cousin", "family"],
        "friend": ["friend", "friends"],
        "loss": ["died", "death", "lost", "passed away"],
        "achievement": ["won", "first", "achievement", "proud", "passed", "success"],
        "memory": ["remember", "memory", "photos", "hometown", "childhood"],
        "fear": ["afraid", "scared", "terrified", "dark", "panic"],
        "tornado": ["tornado"],
        "travel": ["traveling", "driving", "directions", "new house"],
        "layoff": ["layoff", "lay off", "laid off"],
        "pregnancy": ["pregnant", "pregnancy", "children", "baby"],
        "substance": ["drink", "drank", "drugs", "drug", "alcohol"],
        "scary_media": ["scary", "horror", "scary stories", "scary movie"],
        "exercise": ["work out", "workout", "exercise", "gym"],
        "isolation": ["lonely", "alone", "isolated", "don't know anyone", "dont know anyone"],
    }

    response_context_lines = [
        "work pressure",
        "working hard",
        "real relief",
        "something to look forward to",
        "work and rejection",
        "school pressure",
        "school wins",
        "relationship pain",
        "family situations",
        "warm family moments",
        "friendship moments",
        "moments with a close friend",
        "friendship conflict",
        "losing someone",
        "health scares",
        "someone you care about is hurt",
        "being there and caring",
        "moments of progress",
        "effort lead to something good",
        "progress feels different",
        "memories like that",
        "some memories",
        "fear can take over",
        "do not feel in control",
        "new place without close support",
        "find your way at night",
        "job news",
        "taken years for you",
        "mind-altering",
        "frightening scene",
        "said you would do",
    ]

    has_user_context = any(
        any(marker in user_text for marker in markers)
        for markers in context_markers.values()
    )
    has_response_context = any(line in response_text for line in response_context_lines)

    if has_user_context:
        return has_response_context

    return has_emotional_validation(response)


def avoids_generic_positivity(response: str) -> bool:
    """Check for common vague positivity phrases."""
    response = normalize(response)
    return not any(marker in response for marker in GENERIC_POSITIVITY_MARKERS)


def sounds_repetitive(response: str) -> bool:
    """Check whether the response follows repeated template wording."""
    response = normalize(response)
    marker_count = sum(marker in response for marker in REPETITIVE_TEMPLATE_MARKERS)
    return marker_count >= 2


def sounds_natural(response: str) -> bool:
    """Approximate naturalness with length and template checks."""
    word_count = len(response.split())
    return 25 <= word_count <= 95 and not sounds_repetitive(response)


def quality_comment(user_message: str, response: str) -> str:
    """Create a short quality comment for one response."""
    checks = {
        "validation": has_emotional_validation(response),
        "contextual": is_contextual(user_message, response),
        "avoids_generic_positivity": avoids_generic_positivity(response),
        "not_repetitive": not sounds_repetitive(response),
        "natural": sounds_natural(response),
    }

    passed = [name for name, value in checks.items() if value]
    failed = [name for name, value in checks.items() if not value]

    if not failed:
        return "Good: passes all lightweight quality checks."

    return (
        f"Needs work: passes {', '.join(passed)}; "
        f"fails {', '.join(failed)}."
    )


def main() -> None:
    """Print 10 sampled responses with quality comments."""
    rows = load_rows()
    random.seed(RANDOM_SEED)
    samples = random.sample(rows, k=min(SAMPLE_COUNT, len(rows)))

    print_section("Ten Regenerated Assistant Responses")
    for index, row in enumerate(samples, start=1):
        response = row["assistant_response"]
        user_message = row["user_message"]

        print(f"\nSample {index} | Line {row['line_number']}")
        print(f"User: {user_message}")
        print(f"Assistant: {response}")
        print(f"Comment: {quality_comment(user_message, response)}")


if __name__ == "__main__":
    main()
