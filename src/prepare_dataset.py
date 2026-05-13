"""Prepare EmpatheticDialogues for Mistral fine-tuning.

This script downloads the EmpatheticDialogues dataset from Hugging Face and
uses it for emotion/context extraction. The original dialogue continuation is
not used as the assistant label because it can sound like another person adding
their own story. Instead, this script generates supportive validation labels
from the user's emotion and situation.
"""

import json
import random
import re
from typing import Any

import pandas as pd

from config import SMALL_TRAINING_SAMPLE_SIZE, TRAIN_SMALL_PATH
from utils import create_folder, print_section


SYSTEM_MESSAGE = (
    "You are an emotional support chatbot. Provide reassurance, emotional "
    "support, and validation. Understand the user's feeling first, then respond "
    "clearly and specifically. Do not diagnose, treat, or replace professional "
    "mental health care. If the user expresses self-harm or immediate danger, "
    "encourage them to contact emergency services or a trusted person immediately."
)


TEXT_REPLACEMENTS = {
    "_comma_": ",",
    "_period_": ".",
    "_question_": "?",
    "_exclamation_": "!",
    "_apostrophe_": "'",
    "_semicolon_": ";",
    "_colon_": ":",
    "_dash_": "-",
}

BANNED_RESPONSE_PHRASES = [
    "what happened is worth paying attention to",
    "that is a lot to take in",
    "start with the part that feels hardest",
    "be a little patient with yourself",
    "it makes sense that",
    "from what you shared",
    "i can see why this stayed with you",
    "what part of it do you want to hold onto most",
    "what part feels most important to hold onto",
    "try giving yourself a pause before reacting to it",
    "try pausing before",
    "take one slow breath",
    "one steady breath",
    "what happened",
    "the part about what happened",
    "this stayed with you",
    "the part about",
]

EMOTION_WORDING = {
    "afraid": "can feel frightening",
    "angry": "can leave anger behind",
    "annoyed": "can be frustrating",
    "anticipating": "can keep your mind busy",
    "anxious": "can feel intimidating",
    "apprehensive": "can make it hard to feel settled",
    "ashamed": "can bring up shame",
    "caring": "shows how much you care",
    "confident": "can feel steadying",
    "content": "can feel calming",
    "devastated": "can hurt deeply",
    "disappointed": "can sting",
    "disgusted": "can feel upsetting",
    "embarrassed": "can feel embarrassing even after the moment passes",
    "excited": "can feel exciting",
    "faithful": "shows how much loyalty matters to you",
    "furious": "can bring up a lot of anger",
    "grateful": "can feel meaningful",
    "guilty": "can bring up guilt",
    "hopeful": "can give you some hope",
    "impressed": "can leave a strong impression",
    "jealous": "can bring up jealousy",
    "joyful": "can feel genuinely joyful",
    "lonely": "can feel lonely",
    "nostalgic": "can bring back a lot of memories",
    "prepared": "can feel reassuring",
    "proud": "can make you feel proud",
    "sad": "can feel painful",
    "sentimental": "can feel meaningful",
    "surprised": "can catch you off guard",
    "terrified": "can feel terrifying",
    "trusting": "shows how important trust is to you",
}

DEFAULT_EMOTION_WORDING = "can bring up a strong reaction"

POSITIVE_EMOTIONS = {
    "anticipating",
    "caring",
    "confident",
    "content",
    "excited",
    "faithful",
    "grateful",
    "hopeful",
    "impressed",
    "joyful",
    "prepared",
    "proud",
    "sentimental",
    "surprised",
    "trusting",
}

DIFFICULT_EMOTIONS = {
    "afraid",
    "angry",
    "annoyed",
    "anxious",
    "apprehensive",
    "ashamed",
    "devastated",
    "disappointed",
    "disgusted",
    "embarrassed",
    "furious",
    "guilty",
    "jealous",
    "lonely",
    "sad",
    "terrified",
}

SELF_HARM_TERMS = [
    "hurt myself",
    "harm myself",
    "kill myself",
    "suicide",
    "end my life",
    "don't want to live",
    "do not want to live",
    "self harm",
    "self-harm",
]

def clean_text(value: Any) -> str:
    """Convert a dataset value into clean text."""
    if value is None:
        return ""

    text = str(value)
    for old_text, new_text in TEXT_REPLACEMENTS.items():
        text = text.replace(old_text, new_text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_useful_response(response: str, minimum_words: int = 5) -> bool:
    """Return True when the response is not empty or too short."""
    return len(response.split()) >= minimum_words


def has_enough_context(situation: str, minimum_words: int = 6) -> bool:
    """Return True when the user situation has enough context for support."""
    return len(situation.split()) >= minimum_words


def normalize_for_comparison(text: str) -> str:
    """Normalize text so weak reply checks are more reliable."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def has_word(text: str, word: str) -> bool:
    """Check for a full word so 'son' does not match 'poisoned'."""
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def has_any_word(text: str, words: list[str]) -> bool:
    """Return True when any full word appears in normalized text."""
    return any(has_word(text, word) for word in words)


def is_duplicate_response(user_message: str, assistant_response: str) -> bool:
    """Return True when the assistant mostly repeats the user message."""
    user_text = normalize_for_comparison(user_message)
    response_text = normalize_for_comparison(assistant_response)

    if not user_text or not response_text:
        return False

    return response_text in user_text or user_text in response_text


def choose_by_index(options: list[str], situation: str) -> str:
    """Choose a stable response line so examples are varied but repeatable."""
    if not options:
        return ""
    return options[len(situation) % len(options)]


def extract_context_fragment(situation: str, max_words: int = 9) -> str:
    """Use only a literal piece of the real situation without adding details."""
    situation = situation.strip().rstrip(".!?")
    situation = re.split(r"[.!?]", situation)[0]
    situation = re.sub(r"^(remember|when)\s+", "", situation, flags=re.IGNORECASE)
    situation = re.sub(r"\bone time\s+", "", situation, flags=re.IGNORECASE)
    situation = re.sub(r"\bi am\b", "you are", situation, flags=re.IGNORECASE)
    situation = re.sub(r"\bi'm\b", "you are", situation, flags=re.IGNORECASE)
    situation = re.sub(r"\bi have\b", "you have", situation, flags=re.IGNORECASE)
    situation = re.sub(r"\bi've\b", "you have", situation, flags=re.IGNORECASE)
    situation = re.sub(r"\bi was\b", "you were", situation, flags=re.IGNORECASE)
    situation = re.sub(r"\bi had\b", "you had", situation, flags=re.IGNORECASE)
    situation = re.sub(r"\bi met\b", "meeting", situation, flags=re.IGNORECASE)
    situation = re.sub(r"\bi\b", "you", situation, flags=re.IGNORECASE)
    situation = re.sub(r"\bmy\b", "your", situation, flags=re.IGNORECASE)
    situation = re.sub(r"\bme\b", "you", situation, flags=re.IGNORECASE)
    situation = re.sub(r"\s+", " ", situation).strip(" ,")
    situation = re.sub(r"^(was|were|am|felt|feel)\s+", "", situation, flags=re.IGNORECASE)
    words = situation.split()
    if len(words) < 4:
        return ""

    selected_words = words if len(words) <= max_words else words[:max_words]
    weak_endings = {"a", "an", "and", "at", "by", "for", "in", "of", "the", "to", "with", "your"}
    while selected_words and selected_words[-1].lower().strip(".,!?") in weak_endings:
        selected_words = selected_words[:-1]

    if len(selected_words) < 4:
        return ""

    return " ".join(selected_words).rstrip(".,!?")


def clean_sentence(text: str) -> str:
    """Normalize punctuation after composing a response."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\.{2,}", ".", text)
    text = text.replace("?.", "?").replace("!.", "!")
    return text


def sentence_start(text: str) -> str:
    """Uppercase the first character without lowercasing proper names."""
    if not text:
        return text
    return text[0].upper() + text[1:]


def ensure_minimum_response_length(response: str, context_reference: str = "") -> str:
    """Add a simple supportive sentence when a label is too short."""
    response = clean_sentence(response)
    if len(response.split()) >= 8:
        return response

    options = [
        "Give yourself a moment with that feeling.",
        "There is no need to rush your reaction.",
        "A small pause can help you steady yourself.",
        "Let yourself handle the next piece slowly.",
    ]
    seed_text = context_reference or response
    support_line = choose_by_index(options, seed_text)
    return clean_sentence(f"{response} {support_line}")


def contains_banned_phrase(text: str) -> bool:
    """Return True when generated text includes banned repeated wording."""
    lowered_text = text.lower()
    return any(phrase in lowered_text for phrase in BANNED_RESPONSE_PHRASES)


def contains_self_harm(text: str) -> bool:
    """Keep crisis language out of normal fine-tuning labels."""
    lowered_text = text.lower()
    return any(term in lowered_text for term in SELF_HARM_TERMS)


def rewrite_if_banned(response: str, context_reference: str, emotion_phrase: str) -> str:
    """Rewrite labels that accidentally contain banned repeated phrases."""
    response = clean_sentence(response)
    if not contains_banned_phrase(response):
        return ensure_minimum_response_length(response, context_reference)

    rewritten = (
        f"{sentence_start(context_reference)} {emotion_phrase}. "
        "A small pause can help before deciding what to do next."
    )
    return ensure_minimum_response_length(rewritten, context_reference)


def build_context_reference(situation: str) -> str:
    """Create a short natural reference to the user's actual situation."""
    text = normalize_for_comparison(situation)

    if contains_self_harm(situation):
        return ""
    if "starting something new" in text:
        return "starting something unfamiliar"
    if has_word(text, "friend") and has_word(text, "betrayed") and has_word(text, "trust"):
        return "being hurt by someone you trusted"
    if "seperated from my wife" in text or "separated from my wife" in text:
        return "being separated and worried about being alone"
    if "wife is expecting a baby" in text:
        return "expecting a baby and feeling the responsibility"
    if "kicked out of house" in text and "without wife and child" in text:
        return "possibly losing your home and living on your own"
    if "boyfriend cheated" in text:
        return "your boyfriend cheating while you were away"
    if "donate my change" in text:
        return "donating your change at the local store"
    if "house car food water" in text or "clean water to drink" in text:
        return "feeling grateful for having basic needs met"
    if "move back to our hometown" in text:
        return "moving back to your hometown after years away"
    if "birthday next year" in text and "cruise" in text:
        return "looking forward to taking a birthday cruise"
    if "didnt get the last job" in text or "didn't get the last job" in text:
        return "not getting the job you applied for"
    if "relaxing with my kids" in text:
        return "relaxing with your kids after a long day"
    if "playlist" in text and "old song" in text:
        return "hearing an old song from childhood"
    if "doctor" in text and "lump" in text:
        return "getting a lump checked by the doctor"
    if "online dating" in text and "pof" in text:
        return "meeting someone from online dating who surprised you"
    if has_word(text, "sister") and has_word(text, "stole") and "from me" in text:
        return "your sister stealing money from you"
    if "car accident" in text and has_any_word(text, ["image", "saw", "seen"]):
        return "seeing the car accident image"
    if "lost my cat" in text or "put my cat to sleep" in text:
        return "losing your cat"
    if "family reunion" in text:
        return "getting together for your family reunion"
    if "deer screamed" in text:
        return "hearing a deer scream during your run"
    if "out of the workforce" in text and has_word(text, "job"):
        return "applying for a job after years out of the workforce"
    if "no one ever visits" in text or "always by myself in this apartment" in text:
        return "being alone in your apartment"
    if has_word(text, "wife") and "work whenever" in text:
        return "wanting the kind of flexibility your wife has"
    if "nurses" in text and "doctors" in text and "hospital" in text:
        return "being cared for by nurses and doctors at the hospital"
    if "first government it job" in text:
        return "starting your first government IT job"
    if "rearranged the garage" in text or "rearrange the garage" in text:
        return "rearranging the garage after putting it off"
    if "manager lied" in text and "bonus" in text:
        return "your manager taking the bonus away"
    if "weeds" in text and "snakes" in text:
        return "your neighbors leaving the weeds so high"
    if "california" in text:
        return "going to California for the first time"
    if "tipped at work" in text:
        return "getting tipped at work"
    if "pokemon" in text and "christmas" in text:
        return "getting those Pokemon games at Christmas"
    if "new car" in text and has_word(text, "parents"):
        return "being surprised with a new car"
    if "birthday cake" in text and "dog" in text:
        return "watching your dog eat the birthday cake"
    if "kids cookies" in text and "blamed" in text and "dog" in text:
        return "eating your kids' cookies and blaming the dog"
    if "being ready for things" in text or "ready for things" in text:
        return "wanting to be ready instead of caught off guard"
    if has_any_word(text, ["yelled", "shouted"]) and has_any_word(text, ["kids", "children"]):
        return "yelling at your kids"
    if "sell" in text and "house" in text:
        return "waiting on your house to sell"
    if "buyer" in text:
        return "finding a possible buyer"
    if "interview" in text:
        return "not getting the interview outcome you wanted"
    if "rejection" in text or "rejected" in text:
        return "getting rejected after trying"
    if "fireworks" in text:
        return "going to the fireworks with your best friend"
    if "didnt have many friends" in text or "didn't have many friends" in text:
        return "not having many friends when you were younger"
    if "warm summer" in text:
        return "looking ahead to a warm summer"
    if "court" in text:
        return "your ex not showing up to court"
    if "harvard" in text:
        return "getting accepted into Harvard"
    if "dog" in text:
        return "how much you care about your dog"
    if "pizza" in text and "kids" in text:
        return "eating the pizza before your kids got up"
    if has_word(text, "shot"):
        return "waiting at the doctor for the shot"
    if "basketball bracket" in text or "$1000" in text:
        return "winning the basketball bracket challenge"
    if "paying rent" in text or "mortgage" in text:
        return "wanting a home of your own"
    if "marathon" in text or "2nd place" in text:
        return "placing in the marathon after training"
    if "puppy" in text:
        return "your brother getting a new puppy"
    if "honor society" in text or "election" in text:
        return "winning the school election"
    if "cheated on my boyfriend" in text:
        return "regretting cheating on your boyfriend"
    if "adopted" in text and "kid" in text:
        return "seeing how much your child has grown"
    if "roommate" in text and "college" in text:
        return "feeling alone at college after your roommate left"
    if has_any_word(text, ["lie", "lied", "lying"]):
        return "finding out someone lied to you"
    if "turning 40" in text:
        return "turning 40 and feeling older"
    if "home alone" in text and "noise" in text:
        return "hearing that noise while you were home alone"
    if "people stared" in text:
        return "people staring after you fell"
    if "slip" in text and "ice" in text:
        return "slipping on the ice at school"
    if "vacation" in text or "trip" in text:
        return "having that trip on your mind"
    if "birthday" in text:
        return "dealing with the birthday plans"
    if "promoted" in text or "promotion" in text:
        return "getting promoted"
    if "hurricane" in text:
        return "preparing your home for the hurricane"
    if "broken in" in text or "broke in" in text:
        return "your home being broken into"
    if "optometrist" in text or "vision" in text or "prescription" in text:
        return "getting worrying news about your vision"
    if "neighbor" in text and "nose" in text:
        return "seeing your neighbor do something so unpleasant"
    if "neighbor" in text and "house is haunted" in text:
        return "seeing ghosts in your neighbor's house"
    if "house key" in text and has_word(text, "neighbor"):
        return "trusting someone with your house key"
    if "left my home" in text and has_word(text, "neighbor"):
        return "trusting someone to look after your home"
    if "spanked" in text and "son" in text:
        return "seeing a child get hurt like that"
    if "grocery store" in text or "purse" in text:
        return "realizing you left your purse at home"
    if "black friday" in text or "tv" in text:
        return "getting the deal you were hoping for"
    if "hospital" in text:
        return "being in the hospital and feeling unsure"
    if "extra hours" in text or "bills" in text:
        return "working extra hours to keep up with bills"
    if "broke" in text and "ankle" in text:
        return "breaking your ankle before the run"
    if has_word(text, "parents"):
        return extract_context_fragment(situation, max_words=12).lower()
    if has_any_word(text, ["mother", "mom"]):
        return "dealing with your mother"
    if has_any_word(text, ["father", "dad"]):
        return "dealing with your father"
    if has_word(text, "daughter"):
        return "thinking about your daughter"
    if has_word(text, "son"):
        return "thinking about your son"
    if has_word(text, "friend") and has_any_word(text, ["ignore", "ignored", "betrayed", "trust"]):
        return "feeling hurt by your friend"
    if has_word(text, "friend"):
        return "dealing with your friend"
    if has_word(text, "girlfriend"):
        return extract_context_fragment(situation, max_words=12).lower()
    if has_word(text, "boyfriend"):
        return "dealing with your boyfriend"
    if has_word(text, "husband"):
        return "dealing with your husband"
    if has_word(text, "wife"):
        return "dealing with your wife"
    if has_word(text, "relationship"):
        return "dealing with the relationship"
    if "interview" in text:
        return "the interview result"
    if has_word(text, "job"):
        return "dealing with the job news"
    if has_any_word(text, ["work", "coworker", "boss"]):
        return "dealing with work"
    if has_word(text, "exam"):
        return "the exam"
    if has_any_word(text, ["school", "college", "class"]):
        return extract_context_fragment(situation, max_words=10).lower()
    if has_any_word(text, ["lonely", "alone", "isolated"]):
        return "feeling alone lately"
    if has_any_word(text, ["scary", "afraid", "terrified", "tornado", "ocean"]):
        return "that frightening moment"
    if has_any_word(text, ["guilty", "lied", "cheated", "stole"]):
        return "carrying guilt about that choice"

    context_reference = extract_context_fragment(situation, max_words=10).lower()
    if not is_safe_context_reference(context_reference):
        return ""
    return context_reference


def is_safe_context_reference(context_reference: str) -> bool:
    """Reject vague or broken context references before label generation."""
    if not context_reference:
        return False
    lowered_text = context_reference.lower()
    blocked_fragments = [
        "what happened",
        "the part about",
        "this stayed with you",
        "situation",
        "you'm",
        "you've",
        "yourself",
    ]
    return not any(fragment in lowered_text for fragment in blocked_fragments)


def choose_style(emotion: str, situation: str) -> int:
    """Rotate response styles in a repeatable random-looking way."""
    seed_text = f"{emotion}|{situation}"
    randomizer = random.Random(seed_text)
    return randomizer.randrange(12)


def build_validated_response(emotion: str, situation: str) -> str:
    """Create a natural support response from the full user context.

    The response uses one of several conversational styles so the model learns
    varied support instead of one predictable template.
    """
    emotion = clean_text(emotion).lower()
    situation = clean_text(situation)
    emotion_phrase = EMOTION_WORDING.get(emotion, DEFAULT_EMOTION_WORDING)
    context_reference = build_context_reference(situation)
    style = choose_style(emotion, situation)
    is_positive = emotion in POSITIVE_EMOTIONS

    if style == 0:
        # Short warm response.
        response = (
            f"{sentence_start(context_reference)} {emotion_phrase}. "
            "That reaction is worth treating with care."
        )
        return rewrite_if_banned(response, context_reference, emotion_phrase)

    if style == 1:
        # Reflective response.
        closing = (
            "Let that good part have its place."
            if is_positive
            else "Handle the next piece slowly."
        )
        response = (
            f"{sentence_start(context_reference)} can bring up a strong reaction. "
            f"{closing}"
        )
        return rewrite_if_banned(response, context_reference, emotion_phrase)

    if style == 2:
        # Gentle statement without a reflective question ending.
        closing = (
            "Let the part that matters to you have some room."
            if is_positive
            else "The hardest part does not need to be solved all at once."
        )
        response = f"{sentence_start(context_reference)} {emotion_phrase}. {closing}"
        return rewrite_if_banned(response, context_reference, emotion_phrase)

    if style == 3:
        # Practical next step.
        next_step = (
            "Let yourself enjoy the good part before moving on."
            if is_positive
            else "Give yourself a moment before deciding how to respond."
        )
        response = f"{sentence_start(context_reference)} {emotion_phrase}. {next_step}"
        return rewrite_if_banned(response, context_reference, emotion_phrase)

    if style == 4:
        # Reassurance.
        reassurance = (
            "You are allowed to let that feel good."
            if is_positive
            else "That reaction deserves honesty and care."
        )
        response = f"{sentence_start(context_reference)} {emotion_phrase}. {reassurance}"
        return rewrite_if_banned(response, context_reference, emotion_phrase)

    if style == 5:
        # Validation without overexplaining.
        response = (
            f"{sentence_start(context_reference)} is worth letting yourself feel. "
            f"It {emotion_phrase}."
            if is_positive
            else (
                f"{sentence_start(context_reference)} is worth taking seriously. "
                f"It {emotion_phrase}."
            )
        )
        return rewrite_if_banned(response, context_reference, emotion_phrase)

    if style == 6:
        # Supportive conversational response.
        response = (
            f"I would not brush off {context_reference}; it {emotion_phrase}. "
            "Give yourself room to respond at your own pace."
        )
        return rewrite_if_banned(response, context_reference, emotion_phrase)

    if style == 7:
        # Calm grounding response.
        grounding = (
            "Let the good part land before you rush past it."
            if is_positive
            else "Focus on the next small thing you can control."
        )
        response = f"{sentence_start(context_reference)} {emotion_phrase}. {grounding}"
        return rewrite_if_banned(response, context_reference, emotion_phrase)

    if style == 8:
        # Direct conversational support.
        response = (
            f"{sentence_start(context_reference)} is a moment you can let yourself appreciate. "
            "You do not have to downplay it."
            if is_positive
            else (
                f"{sentence_start(context_reference)} is enough reason to feel affected. "
                "You can be honest about that without judging yourself."
            )
        )
        return rewrite_if_banned(response, context_reference, emotion_phrase)

    if style == 9:
        # Practical but gentle.
        next_step = (
            "Share it with someone who will be glad for you if that feels right."
            if is_positive
            else "Talking it through with someone steady could help you feel less alone in it."
        )
        response = f"{sentence_start(context_reference)} {emotion_phrase}. {next_step}"
        return rewrite_if_banned(response, context_reference, emotion_phrase)

    if style == 10:
        # Natural reassurance.
        response = (
            f"{sentence_start(context_reference)} is something you can feel good about. "
            "You can let it matter."
            if is_positive
            else (
                f"Anyone dealing with {context_reference} might need a minute. "
                "Take the pressure off having the perfect reaction."
            )
        )
        return rewrite_if_banned(response, context_reference, emotion_phrase)

    # Brief grounded closing.
    grounding = (
        "Let yourself enjoy it without downplaying it."
        if is_positive
        else "Focus on the next small thing you can control."
    )
    response = f"{sentence_start(context_reference)} {emotion_phrase}. {grounding}"
    return rewrite_if_banned(response, context_reference, emotion_phrase)


def build_user_message(example: dict[str, Any]) -> str:
    """Create the user side of the conversation from dataset columns."""
    situation = clean_text(example.get("prompt"))
    emotion = clean_text(example.get("context"))

    if emotion and situation:
        return f"I feel {emotion}. {situation}"
    if situation:
        return situation
    if emotion:
        return f"I feel {emotion}."
    return ""


def build_training_dataframe(dataset: Any) -> pd.DataFrame:
    """Clean the dataset with pandas and build supportive labels.

    EmpatheticDialogues contains real dialogue turns, but the next utterance is
    often a continuation of the conversation rather than the kind of response a
    support chatbot should learn. For this project, the dataset provides the
    emotion and situation; the assistant label is engineered separately.
    """
    dataframe = pd.DataFrame(dataset)

    dataframe["emotion"] = dataframe["context"].apply(clean_text)
    dataframe["situation"] = dataframe["prompt"].apply(clean_text)

    dataframe["user_message"] = dataframe.apply(
        lambda row: build_user_message(
            {
                "context": row["emotion"],
                "prompt": row["situation"],
            }
        ),
        axis=1,
    )

    dataframe = dataframe[
        (dataframe["user_message"].str.len() > 0)
        & (dataframe["situation"].str.len() > 0)
        & (dataframe["emotion"].str.len() > 0)
    ].copy()

    before_context_filter = len(dataframe)
    dataframe = dataframe[
        dataframe["situation"].apply(has_enough_context)
    ].copy()
    print(f"Rows removed for low context: {before_context_filter - len(dataframe)}")

    dataframe["context_reference"] = dataframe["situation"].apply(build_context_reference)
    before_grounding_filter = len(dataframe)
    dataframe = dataframe[
        dataframe["context_reference"].apply(is_safe_context_reference)
    ].copy()
    print(
        "Rows removed without safe concrete grounding: "
        f"{before_grounding_filter - len(dataframe)}"
    )

    before_dedupe = len(dataframe)
    dataframe = dataframe.drop_duplicates(subset=["user_message"]).copy()
    print(f"Duplicate user messages removed: {before_dedupe - len(dataframe)}")

    dataframe["assistant_response"] = dataframe.apply(
        lambda row: build_validated_response(row["emotion"], row["situation"]),
        axis=1,
    )
    dataframe["was_rewritten"] = True

    dataframe = dataframe[
        dataframe["assistant_response"].apply(is_useful_response)
    ].copy()

    dataframe["text"] = dataframe.apply(
        lambda row: build_mistral_chat_text(
            row["user_message"],
            row["assistant_response"],
        ),
        axis=1,
    )

    return dataframe


def build_mistral_chat_text(user_message: str, assistant_response: str) -> str:
    """Convert one example into Mistral instruction-response text."""
    return (
        "<s>[INST] "
        f"<<SYS>>\n{SYSTEM_MESSAGE}\n<</SYS>>\n\n"
        f"{user_message} [/INST] "
        f"{assistant_response}</s>"
    )


def prepare_examples() -> list[dict[str, str]]:
    """Load, filter, format, and return a small training subset."""
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "The 'datasets' library is not installed. Run "
            "'pip install -r requirements.txt' before preparing the dataset."
        ) from error

    # EmpatheticDialogues uses a Hugging Face dataset script, so this flag is
    # required by datasets 3.x to load the official dataset builder.
    dataset = load_dataset(
        "empathetic_dialogues",
        split="train",
        trust_remote_code=True,
    )
    dataframe = build_training_dataframe(dataset)
    sample = dataframe.head(SMALL_TRAINING_SAMPLE_SIZE)

    rewritten_count = int(sample["was_rewritten"].sum())
    print(f"Validated assistant responses generated in sample: {rewritten_count}")

    return sample[
        ["text", "user_message", "assistant_response", "was_rewritten"]
    ].to_dict("records")


def save_jsonl(rows: list[dict[str, str]]) -> None:
    """Save prepared examples as one JSON object per line."""
    create_folder(TRAIN_SMALL_PATH.parent)

    with TRAIN_SMALL_PATH.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """Run the small dataset preparation workflow."""
    print_section("Preparing EmpatheticDialogues small training subset")
    rows = prepare_examples()
    save_jsonl(rows)

    print(f"Saved rows: {len(rows)}")
    print(f"Output file: {TRAIN_SMALL_PATH}")
    if rows:
        print("\nFirst formatted example:")
        print(rows[0]["text"][:1000])


if __name__ == "__main__":
    main()
