"""Create the v3 cleaned local training dataset.

This script does not download or load any Hugging Face dataset. It only reads
the local JSONL file at `data/train_small.jsonl`, rewrites assistant responses
that contain repeated/template phrases, and saves `data/train_clean_v3.jsonl`.
"""

from __future__ import annotations

import json
import random
import re

from config import TRAIN_CLEAN_V3_PATH, TRAIN_SMALL_PATH
from prepare_dataset import (
    SYSTEM_MESSAGE,
    build_context_reference,
    clean_sentence,
    clean_text,
    extract_context_fragment,
)
from utils import create_folder, print_section


BAD_TEMPLATE_PHRASES = [
    "take one slow breath and deal with the next small piece first",
    "try giving yourself a pause before reacting to it",
    "i can see why this stayed with you",
    "what part of it do you want to hold onto most",
    "that is a lot to take in",
    "one steady breath",
    "try pausing before",
    "what part feels most important to hold onto",
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
    "without being hard on yourself",
    "give that reaction some room",
    "let yourself acknowledge it while you decide what comes next",
]

REWRITE_ALL_RESPONSES = True

EMOTION_RESPONSE_PARTS = {
    "afraid": ("honestly", "that would shake a person"),
    "angry": ("yeah", "that kind of thing can be frustrating"),
    "annoyed": ("yeah", "that would get irritating fast"),
    "anticipating": ("honestly", "waiting for it can keep your mind busy"),
    "anxious": ("yeah", "uncertainty like that can get under your skin"),
    "apprehensive": ("honestly", "not knowing how it will go can feel unsettling"),
    "ashamed": ("yeah", "shame can sit heavily after something like that"),
    "caring": ("honestly", "your care for them comes through clearly"),
    "confident": ("honestly", "it is okay to let yourself feel good about it"),
    "content": ("honestly", "having a calmer moment like that can feel steady"),
    "devastated": ("honestly", "that kind of hurt can hit deeply"),
    "disappointed": ("honestly", "that kind of letdown can sting"),
    "disgusted": ("yeah", "that would be hard to be around"),
    "embarrassed": ("yeah", "that can feel awkward even after the moment passes"),
    "excited": ("of course", "that is something you can let yourself enjoy"),
    "faithful": ("honestly", "that loyalty clearly matters to you"),
    "furious": ("yeah", "that would make a lot of people furious"),
    "grateful": ("honestly", "that kind of moment can feel good to hold onto"),
    "guilty": ("yeah", "guilt can weigh on you after something like that"),
    "hopeful": ("honestly", "having some hope there is completely human"),
    "impressed": ("honestly", "it is okay to be impressed by that"),
    "jealous": ("yeah", "jealousy can show up even when you do not want it to"),
    "joyful": ("honestly", "that joy is something you can let yourself have"),
    "lonely": ("yeah", "being alone like that can wear on you"),
    "nostalgic": ("honestly", "memories like that can pull a lot up"),
    "prepared": ("honestly", "feeling ready for it can be reassuring"),
    "proud": ("of course", "that is something to feel good about"),
    "sad": ("honestly", "that can hurt"),
    "sentimental": ("honestly", "moments like that can bring old feelings back"),
    "surprised": ("obviously", "that is unexpected"),
    "terrified": ("honestly", "that would feel scary in the moment"),
    "trusting": ("honestly", "trust matters a lot in a situation like that"),
}

DEFAULT_RESPONSE_PARTS = ("honestly", "your reaction is human")

REWRITE_CLOSINGS = [
    "You should not have to be hard on yourself for reacting to it.",
    "It is okay to need a minute with that.",
    "Let the feeling be there without turning it against yourself.",
    "You can care about it without judging your reaction.",
    "That is a human reaction, not something you have to force away.",
    "You do not need to rush yourself past it.",
    "It is okay if it takes a little time to settle.",
    "You can notice your reaction without making yourself the problem.",
    "Let yourself respond like a person, not like you have to get it perfect.",
    "You are allowed to have a reaction before everything feels clear.",
]

GENERIC_CONTEXT_REFERENCES = [
    "thinking about your daughter",
    "thinking about your son",
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
    "having that trip on your mind",
]


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace for phrase checks."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def normalize_for_duplicate_check(text: str) -> str:
    """Normalize text so duplicate prompts can be removed reliably."""
    text = normalize(text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_bad_template(response: str) -> bool:
    """Return True when a response contains a banned repeated phrase."""
    response = normalize(response)
    return any(phrase in response for phrase in BAD_TEMPLATE_PHRASES)


def contains_broken_context(response: str) -> bool:
    """Detect clipped context fragments produced by earlier cleanup versions."""
    response = normalize(response)
    broken_patterns = [
        r"\bwith your can\b",
        r"\bwith you can\b",
        r"\band an can\b",
        r"\bto the can\b",
        r"\bof can\b",
    ]
    return any(re.search(pattern, response) for pattern in broken_patterns)


def has_context_mismatch(user_message: str, response: str) -> bool:
    """Find known stale labels that reference details missing from the prompt."""
    user_text = normalize(user_message)
    response_text = normalize(response)

    if "carrying guilt about that choice" in response_text:
        return not any(word in user_text for word in ["guilt", "guilty", "cheated", "stole"])

    if "waiting at the doctor for the shot" in response_text:
        return "shot" not in user_text

    if "finding out someone lied to you" in response_text:
        return not any(re.search(rf"\b{word}\b", user_text) for word in ["lie", "lied", "lying"])

    return False


def should_rewrite_response(user_message: str, response: str) -> bool:
    """Return True when a row needs a safer v3 assistant label."""
    return (
        contains_bad_template(response)
        or contains_broken_context(response)
        or has_context_mismatch(user_message, response)
    )


def load_jsonl_rows() -> list[dict[str, str]]:
    """Load the existing local JSONL rows."""
    rows: list[dict[str, str]] = []

    with TRAIN_SMALL_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def split_user_message(user_message: str) -> tuple[str, str]:
    """Extract emotion and situation from the prepared user message."""
    user_message = clean_text(user_message)
    match = re.match(r"^I feel ([^.]+)\.\s*(.+)$", user_message, flags=re.IGNORECASE)

    if not match:
        return "", user_message

    emotion = clean_text(match.group(1)).lower()
    situation = clean_text(match.group(2))
    return emotion, situation


def clean_context_reference(context_reference: str, situation: str) -> str:
    """Avoid generic relationship labels and use the user's concrete event."""
    normalized_reference = normalize(context_reference)

    relation_rewrites = {
        "your parents surprising you with a new car": "being surprised with a new car",
        "you were surprised when your parents bought you a car": "being surprised with a car you did not expect",
        "your parents always taught you the value of patience": "learning patience from your parents",
        "your neighbor's house is haunted, you have seen ghosts": "seeing ghosts in your neighbor's house",
        "is it okay that you give your neighbor your house": "trusting someone with your house key",
        "you left your home to your neighbor to take care of, when": "trusting someone to look after your home",
        "whenever your girlfriend at the time broke up with you and ghosted": "being ghosted after the breakup",
    }

    if normalized_reference in relation_rewrites:
        return relation_rewrites[normalized_reference]

    if normalized_reference.startswith("your spouse and you "):
        context_reference = re.sub(
            r"^your spouse and you\b",
            "you and your spouse",
            context_reference,
            flags=re.IGNORECASE,
        )
    if normalized_reference.startswith("your wife and you "):
        context_reference = re.sub(
            r"^your wife and you\b",
            "you and your wife",
            context_reference,
            flags=re.IGNORECASE,
        )

    if normalized_reference in GENERIC_CONTEXT_REFERENCES:
        literal_fragment = extract_literal_context(situation, max_words=28)
        if literal_fragment:
            literal_fragment = re.sub(
                r"^your wife and you\b",
                "you and your wife",
                literal_fragment,
                flags=re.IGNORECASE,
            )
            literal_fragment = re.sub(
                r"^your spouse and you\b",
                "you and your spouse",
                literal_fragment,
                flags=re.IGNORECASE,
            )
            return literal_fragment

    return context_reference


def extract_literal_context(situation: str, max_words: int = 28) -> str:
    """Extract a concrete situation sentence without stopping at emotion filler."""
    sentence_parts = re.split(r"[.!?]", situation)

    for sentence in sentence_parts:
        sentence = clean_text(sentence)
        lowered = normalize(sentence)

        if len(sentence.split()) < 4:
            continue
        if re.fullmatch(r"i (am|was|feel|felt) (really )?[a-z]+", lowered):
            continue

        fragment = extract_context_fragment(sentence, max_words=max_words)
        if fragment:
            return fragment

    return extract_context_fragment(situation, max_words=max_words)


def looks_incomplete_context(context_reference: str) -> bool:
    """Detect context fragments that were cut off in the middle of an idea."""
    words = normalize(context_reference).split()
    if len(words) < 4:
        return True

    weak_endings = {
        "a",
        "about",
        "after",
        "an",
        "and",
        "are",
        "at",
        "back",
        "been",
        "boyfriend",
        "by",
        "came",
        "due",
        "father",
        "for",
        "from",
        "friend",
        "girlfriend",
        "had",
        "has",
        "husband",
        "in",
        "mother",
        "of",
        "our",
        "parent",
        "parents",
        "beginning",
        "that",
        "the",
        "to",
        "you",
        "was",
        "were",
        "when",
        "wife",
        "with",
        "year",
        "we",
    }
    return words[-1] in weak_endings


def choose_context_reference(situation: str) -> str:
    """Prefer specific context, but fall back to a longer literal fragment."""
    context_reference = build_context_reference(situation)
    context_reference = clean_context_reference(context_reference, situation)

    if not context_reference or looks_incomplete_context(context_reference):
        for max_words in (16, 28, 40):
            literal_fragment = extract_literal_context(situation, max_words=max_words)
            if literal_fragment:
                context_reference = clean_context_reference(literal_fragment, situation)
            if context_reference and not looks_incomplete_context(context_reference):
                break

    if not context_reference:
        context_reference = situation.rstrip(".!?")

    return context_reference.rstrip(" ,.!?")


def capitalize_first(text: str) -> str:
    """Capitalize only the first character of a generated sentence."""
    if not text:
        return text
    return text[0].upper() + text[1:]


def connector_clause(middle_clause: str) -> str:
    """Make the emotion phrase fit after a comma and 'and'."""
    if middle_clause.startswith("that is "):
        return f"it is {middle_clause.removeprefix('that is ')}"
    if middle_clause.startswith("that would "):
        return f"it would {middle_clause.removeprefix('that would ')}"
    return middle_clause


def build_first_sentence(
    opener: str,
    context_reference: str,
    middle_clause: str,
    randomizer: random.Random,
) -> str:
    """Create one conversational sentence that keeps the user's real detail."""
    context_reference = context_reference.rstrip(" ,.!?")
    lower_context = context_reference.lower()
    connected_middle = connector_clause(middle_clause)

    action_starts = (
        "being ",
        "donating ",
        "expecting ",
        "feeling ",
        "finding ",
        "getting ",
        "having ",
        "hearing ",
        "learning ",
        "looking ",
        "meeting ",
        "moving ",
        "not getting ",
        "relaxing ",
        "seeing ",
        "starting ",
        "trusting ",
        "waiting ",
        "wanting ",
    )
    if lower_context.startswith(action_starts):
        if middle_clause.startswith("that is "):
            return f"{opener.capitalize()}, {context_reference} is {middle_clause.removeprefix('that is ')}."
        if middle_clause.startswith("that would "):
            return f"{opener.capitalize()}, {context_reference} would {middle_clause.removeprefix('that would ')}."
        return f"{opener.capitalize()}, {context_reference}, and {connected_middle}."

    if lower_context.startswith(("a ", "an ", "the ")):
        return f"{opener.capitalize()}, with {context_reference}, {connected_middle}."

    sentence_shapes = [
        f"{opener.capitalize()}, {context_reference}, and {connected_middle}.",
        f"{opener.capitalize()}, {context_reference}. {capitalize_first(connected_middle)}.",
    ]
    return randomizer.choice(sentence_shapes)


def build_clean_response(user_message: str) -> str:
    """Build a natural replacement response for a bad/template label."""
    emotion, situation = split_user_message(user_message)
    context_reference = choose_context_reference(situation)
    opener, middle_clause = EMOTION_RESPONSE_PARTS.get(emotion, DEFAULT_RESPONSE_PARTS)
    randomizer = random.Random(user_message)
    closing = randomizer.choice(REWRITE_CLOSINGS)

    first_sentence = build_first_sentence(
        opener,
        context_reference,
        middle_clause,
        randomizer,
    )
    response = f"{first_sentence} {closing}"
    return clean_sentence(response)


def build_mistral_chat_text(user_message: str, assistant_response: str) -> str:
    """Rebuild the Mistral training text after rewriting the response."""
    return (
        "<s>[INST] "
        f"<<SYS>>\n{SYSTEM_MESSAGE}\n<</SYS>>\n\n"
        f"{user_message} [/INST] "
        f"{assistant_response}</s>"
    )


def clean_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int, int, int]:
    """Rewrite bad templates and remove duplicate prompts/responses."""
    cleaned_rows: list[dict[str, str]] = []
    rewritten_count = 0
    duplicate_prompt_count = 0
    duplicate_response_count = 0
    seen_user_messages: set[str] = set()
    seen_assistant_responses: set[str] = set()

    for row in rows:
        cleaned_row = dict(row)
        user_message = cleaned_row.get("user_message", "")
        duplicate_key = normalize_for_duplicate_check(user_message)

        if duplicate_key in seen_user_messages:
            duplicate_prompt_count += 1
            continue
        seen_user_messages.add(duplicate_key)

        assistant_response = cleaned_row.get("assistant_response", "")

        if REWRITE_ALL_RESPONSES or should_rewrite_response(user_message, assistant_response):
            assistant_response = build_clean_response(user_message)
            cleaned_row["assistant_response"] = assistant_response
            cleaned_row["text"] = build_mistral_chat_text(user_message, assistant_response)
            cleaned_row["was_rewritten"] = True
            rewritten_count += 1

        response_key = normalize_for_duplicate_check(assistant_response)
        if response_key in seen_assistant_responses:
            duplicate_response_count += 1
            continue
        seen_assistant_responses.add(response_key)

        cleaned_rows.append(cleaned_row)

    return cleaned_rows, rewritten_count, duplicate_prompt_count, duplicate_response_count


def save_jsonl(rows: list[dict[str, str]]) -> None:
    """Save the cleaned v3 dataset as JSONL."""
    create_folder(TRAIN_CLEAN_V3_PATH.parent)

    with TRAIN_CLEAN_V3_PATH.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """Run the local-only v3 dataset cleanup."""
    print_section("Cleaning Local Dataset v3")
    print(f"Input file: {TRAIN_SMALL_PATH}")
    print(f"Output file: {TRAIN_CLEAN_V3_PATH}")

    if not TRAIN_SMALL_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {TRAIN_SMALL_PATH}\n"
            "Run `python src/prepare_dataset.py` first."
        )

    rows = load_jsonl_rows()
    (
        cleaned_rows,
        rewritten_count,
        duplicate_prompt_count,
        duplicate_response_count,
    ) = clean_rows(rows)
    save_jsonl(cleaned_rows)

    print(f"Rows read: {len(rows)}")
    print(f"Rows written: {len(cleaned_rows)}")
    print(f"Rows rewritten: {rewritten_count}")
    print(f"Duplicate user-message rows removed: {duplicate_prompt_count}")
    print(f"Duplicate assistant-response rows removed: {duplicate_response_count}")


if __name__ == "__main__":
    main()
