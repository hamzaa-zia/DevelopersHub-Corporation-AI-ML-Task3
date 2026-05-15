"""CPU-safe demo response engine for the Streamlit chatbot app.

The real project trains and loads a Mistral 7B LoRA adapter, but that model is
too heavy for normal CPU-only review. This file keeps the demo app usable by
returning safe, varied, rule-based support responses without loading a GPU
model.
"""

from __future__ import annotations

import re

from safety import get_safe_response


RESPONSE_RULES = [
    {
        "label": "accident",
        "keywords": [
            "accident",
            "near death",
            "almost died",
            "crash",
            "hit by",
            "trauma",
            "traumatic",
            "scared",
            "terrified",
        ],
        "responses": [
            (
                "That accident sounds really frightening, especially because it "
                "felt close to death. It makes sense if your mind is still replaying "
                "it; try to stay near someone you trust today and give yourself time "
                "to feel steady again."
            ),
            (
                "A near-death experience can leave your body and mind on high alert "
                "even after the danger has passed. Be gentle with yourself, avoid "
                "rushing back to normal, and consider talking it through with someone "
                "safe."
            ),
        ],
    },
    {
        "label": "academic_setback",
        "keywords": [
            "failed",
            "test",
            "exam",
            "quiz",
            "university",
            "college",
            "assignment",
            "grade",
            "result",
        ],
        "responses": [
            (
                "Failing a test can feel heavy, but it does not make you a failure. "
                "Give yourself a little time to settle, then look at what went wrong "
                "and choose one topic to review first."
            ),
            (
                "That university result sounds disappointing, and it is understandable "
                "to feel discouraged. One bad test is feedback about preparation or "
                "strategy, not a final judgment on your ability."
            ),
        ],
    },
    {
        "label": "positive_friendship",
        "keywords": [
            "gift",
            "surprised me",
            "surprise",
            "friend surprised",
            "happy",
            "excited",
            "grateful",
            "thankful",
        ],
        "responses": [
            (
                "That sounds like a genuinely warm moment. Let yourself enjoy the gift "
                "and the care behind it; small surprises from friends can mean a lot."
            ),
            (
                "Being surprised by a friend can feel really special. It is okay to "
                "hold onto that good feeling and let them know the gesture mattered."
            ),
        ],
    },
    {
        "label": "loneliness",
        "keywords": ["lonely", "alone", "isolated", "no one", "nobody", "left out"],
        "responses": [
            (
                "Feeling lonely can be painful, especially when it seems like no one "
                "notices. If possible, send one simple message to a safe person today, "
                "even if you only say you could use someone to talk to."
            ),
            (
                "That kind of loneliness can make everything feel heavier. You do not "
                "need a perfect conversation right now; one small connection can be a "
                "good first step."
            ),
        ],
    },
    {
        "label": "anxiety",
        "keywords": [
            "anxious",
            "anxiety",
            "worried",
            "worry",
            "nervous",
            "panic",
            "overthinking",
        ],
        "responses": [
            (
                "That worry sounds exhausting. Try naming the exact thing you are "
                "afraid might happen, then write one practical step you can take next."
            ),
            (
                "Anxiety can make the future feel much bigger than the present. Bring "
                "your focus back to the next few minutes and handle only the next "
                "small piece."
            ),
        ],
    },
    {
        "label": "anger",
        "keywords": ["angry", "furious", "mad", "annoyed", "irritated", "betrayed"],
        "responses": [
            (
                "Your anger makes sense if something felt unfair or hurtful. Before "
                "you respond, give yourself a short pause so you can choose words that "
                "protect you instead of escalating the situation."
            ),
            (
                "Feeling angry does not make you wrong; it usually means something "
                "mattered. Let the first wave pass, then decide what boundary or "
                "conversation is actually needed."
            ),
        ],
    },
    {
        "label": "guilt",
        "keywords": ["guilty", "regret", "ashamed", "mistake", "my fault"],
        "responses": [
            (
                "Guilt can feel intense when you care about what happened. Try to "
                "separate what you can repair from what you are only using to punish "
                "yourself."
            ),
            (
                "Regret is painful, but it can also point toward what matters to you. "
                "If there is a repair you can make, start there; if not, focus on what "
                "you can learn without attacking yourself."
            ),
        ],
    },
    {
        "label": "work_pressure",
        "keywords": ["work", "job", "boss", "coworker", "deadline", "interview"],
        "responses": [
            (
                "Work pressure can pile up quickly and make everything feel urgent. "
                "Pick the one task or conversation that matters most first, then give "
                "your attention to that instead of the whole pile."
            ),
            (
                "That work situation sounds stressful. You deserve a clear next step, "
                "so start by writing what is urgent, what can wait, and what support "
                "you can ask for."
            ),
        ],
    },
    {
        "label": "family",
        "keywords": ["family", "parents", "mother", "father", "sister", "brother"],
        "responses": [
            (
                "Family situations can hurt deeply because they are so personal. Try "
                "to name what you need most right now: space, a calm conversation, or "
                "support from someone outside the conflict."
            ),
            (
                "That sounds emotionally tiring, especially because family problems "
                "can follow you around. Give yourself permission to step back before "
                "trying to explain everything."
            ),
        ],
    },
    {
        "label": "health",
        "keywords": ["sick", "hospital", "doctor", "pain", "injury", "injured", "ill"],
        "responses": [
            (
                "Health worries can make you feel vulnerable and unsettled. If the "
                "symptoms or injury are serious, please contact a qualified medical "
                "professional or someone nearby who can help."
            ),
            (
                "It is understandable to feel shaken when your health is involved. "
                "Stay close to practical support, follow medical guidance if you have "
                "it, and do not handle it completely alone."
            ),
        ],
    },
]

FALLBACK_RESPONSES = [
    (
        "That sounds like a lot to carry. Your reaction is valid, and you can "
        "start by naming the hardest part of the situation before deciding what "
        "to do next."
    ),
    (
        "I can understand why that would affect you. Give yourself a moment to "
        "slow down, then choose one small step that would make the situation feel "
        "slightly more manageable."
    ),
    (
        "What you shared sounds important, and it deserves care instead of self-"
        "judgment. Try to focus on what you need in the next hour, not the whole "
        "problem at once."
    ),
]


def normalize_text(text: str) -> str:
    """Lowercase text and collapse extra spaces for reliable keyword matching."""
    lowered_text = text.lower()
    return re.sub(r"\s+", " ", lowered_text).strip()


def contains_keyword(text: str, keywords: list[str]) -> bool:
    """Return True when a rule keyword or phrase appears in the user message."""
    return any(keyword in text for keyword in keywords)


def choose_response(options: list[str], user_text: str) -> str:
    """Choose a repeatable response variant so similar prompts are not identical."""
    index = sum(ord(character) for character in user_text) % len(options)
    return options[index]


def find_matching_rule(user_text: str) -> dict[str, object] | None:
    """Find the first response rule that matches the user's message."""
    normalized_text = normalize_text(user_text)

    for rule in RESPONSE_RULES:
        keywords = rule["keywords"]
        if contains_keyword(normalized_text, keywords):
            return rule

    return None


def get_demo_keyword_summary() -> dict[str, list[str]]:
    """Return keyword categories shown in the Streamlit demo explanation."""
    return {
        str(rule["label"]).replace("_", " ").title(): list(rule["keywords"])
        for rule in RESPONSE_RULES
    }


def generate_demo_response(user_text: str) -> str:
    """Return a safe supportive response without loading the fine-tuned model."""
    safe_response = get_safe_response(user_text)
    if safe_response:
        return safe_response

    matching_rule = find_matching_rule(user_text)
    if matching_rule:
        return choose_response(matching_rule["responses"], user_text)

    return choose_response(FALLBACK_RESPONSES, user_text)
