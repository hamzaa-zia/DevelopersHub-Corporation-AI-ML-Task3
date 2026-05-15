"""Basic safety rules for chatbot responses.

This file keeps crisis handling separate from model inference. The chatbot
should never diagnose a user and should never claim that it replaces therapy.
It should understand the user's message first, then respond with validation,
reassurance, and specific emotional support.
"""


CRISIS_KEYWORDS = [
    "cut myself",
    "cutting myself",
    "hurt myself",
    "hurting myself",
    "harm myself",
    "harming myself",
    "kill myself",
    "killing myself",
    "suicide",
    "suicidal",
    "end my life",
    "don't want to live",
    "do not want to live",
    "cannot go on",
    "can't go on",
    "overdose",
    "self harm",
    "self-harm",
]


CRISIS_RESPONSE = (
    "[SAFETY RESPONSE TRIGGERED]\n"
    "I'm really sorry you're feeling this way. If you may hurt yourself or are "
    "in immediate danger, please contact emergency services or a trusted person "
    "right now. You do not have to handle this alone."
)


def contains_crisis_language(user_text: str) -> bool:
    """Check whether the user message contains self-harm crisis language."""
    normalized_text = user_text.lower()
    return any(keyword in normalized_text for keyword in CRISIS_KEYWORDS)


def get_safe_response(user_text: str) -> str | None:
    """Return a crisis response when needed, otherwise return None."""
    if contains_crisis_language(user_text):
        return CRISIS_RESPONSE
    return None
