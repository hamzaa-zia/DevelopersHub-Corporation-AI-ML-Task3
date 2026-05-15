"""Streamlit interface for the mental health support chatbot.

Run with:
    streamlit run app.py

The interface supports two modes:
1. CPU-safe keyword demo mode for normal local review.
2. Full Mistral 7B LoRA adapter mode for Colab/CUDA GPU testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


# Make the local `src` folder importable when Streamlit runs from the project
# root. This keeps the app simple and avoids packaging steps for a submission.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from demo_chatbot import generate_demo_response, get_demo_keyword_summary  # noqa: E402
from safety import get_safe_response  # noqa: E402


APP_TITLE = "Mental Health Support Chatbot"
DEMO_MODE = "Keyword Demo (CPU)"
FULL_MODEL_MODE = "Full Mistral Adapter (GPU)"
INITIAL_ASSISTANT_MESSAGE = (
    "Hi, I am here to offer emotional support. Share what you are feeling or "
    "what happened, and I will respond supportively within safe limits."
)


def initialize_chat_history() -> None:
    """Create chat history once so messages stay visible after each response."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": INITIAL_ASSISTANT_MESSAGE,
            }
        ]


def reset_chat_if_mode_changed(selected_mode: str) -> None:
    """Clear old chat messages when switching between demo and real model modes."""
    if st.session_state.get("selected_mode") == selected_mode:
        return

    st.session_state.selected_mode = selected_mode
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": INITIAL_ASSISTANT_MESSAGE,
        }
    ]


def render_sidebar() -> str:
    """Show mode controls, project limits, and safety context."""
    with st.sidebar:
        st.title("Chatbot Mode")
        selected_mode = st.radio(
            "Choose response engine",
            [DEMO_MODE, FULL_MODEL_MODE],
            help=(
                "Use Keyword Demo on CPU. Use Full Mistral Adapter only on "
                "Colab GPU or a CUDA-capable machine."
            ),
        )

        if selected_mode == DEMO_MODE:
            st.error(
                "Demo mode is CPU-safe and keyword-based. It does not load the "
                "real Mistral 7B LoRA adapter."
            )
        else:
            st.warning(
                "Full adapter mode loads Mistral 7B with the LoRA adapter from "
                "`src/inference.py`. Use this only on Colab GPU or CUDA GPU."
            )

        st.divider()
        st.markdown("**Safety limits**")
        st.write(
            "The chatbot provides emotional support only. It does not diagnose, "
            "treat, or replace therapy, counseling, emergency care, or medical "
            "advice."
        )
        st.write(
            "For self-harm or immediate danger messages, the app returns a "
            "fixed crisis-safety response instead of generating a normal reply."
        )
        return selected_mode


def render_demo_notice() -> None:
    """Explain clearly that the visible chatbot is a keyword-based demo."""
    st.error(
        "Demo alert: this interface is for demonstration purposes only. It does "
        "not use the trained Mistral 7B adapter during local review; it uses "
        "keyword categories to return safe sample responses."
    )

    with st.expander("View demo keyword categories"):
        keyword_summary = get_demo_keyword_summary()
        for category, keywords in keyword_summary.items():
            st.markdown(f"**{category}:** {', '.join(keywords)}")


def render_full_model_notice() -> None:
    """Explain clearly when the interface is using the real adapter path."""
    st.warning(
        "Full adapter mode is selected. The first response can take time because "
        "the app loads Mistral 7B and attaches the LoRA adapter. This requires "
        "GPU memory and will not run properly on a normal CPU-only machine."
    )


@st.cache_resource(show_spinner=False)
def load_real_chatbot():
    """Load the tokenizer and LoRA-attached model once for Streamlit."""
    from inference import load_model_with_adapter, load_tokenizer

    tokenizer = load_tokenizer()
    model = load_model_with_adapter()
    return tokenizer, model


def generate_real_model_response(user_message: str) -> str:
    """Generate a response with the real Mistral adapter when GPU is available."""
    safe_response = get_safe_response(user_message)
    if safe_response:
        return safe_response

    from inference import generate_response

    tokenizer, model = load_real_chatbot()
    return generate_response(user_message, model, tokenizer)


def generate_response_for_mode(user_message: str, selected_mode: str) -> str:
    """Route the user message through the selected response engine."""
    if selected_mode == FULL_MODEL_MODE:
        return generate_real_model_response(user_message)

    return generate_demo_response(user_message)


def render_chat_history() -> None:
    """Display every previous user and assistant message as chat bubbles."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def handle_user_message(user_message: str, selected_mode: str) -> None:
    """Save the user message, generate a response, and save the assistant reply."""
    st.session_state.messages.append({"role": "user", "content": user_message})

    with st.chat_message("user"):
        st.write(user_message)

    try:
        if selected_mode == FULL_MODEL_MODE:
            with st.spinner("Loading/generating with the Mistral LoRA adapter..."):
                assistant_response = generate_response_for_mode(user_message, selected_mode)
        else:
            assistant_response = generate_response_for_mode(user_message, selected_mode)
    except Exception as error:
        assistant_response = (
            "Full Mistral adapter mode could not run in this environment. "
            "Use Colab GPU/CUDA GPU and confirm the adapter files are available. "
            f"Technical reason: {error}"
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response}
    )

    with st.chat_message("assistant"):
        st.write(assistant_response)


def main() -> None:
    """Run the Streamlit chatbot page."""
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    initialize_chat_history()

    st.title(APP_TITLE)
    st.caption("Streamlit interface for demo mode and GPU-based adapter testing")

    selected_mode = render_sidebar()
    reset_chat_if_mode_changed(selected_mode)

    if selected_mode == DEMO_MODE:
        render_demo_notice()
    else:
        render_full_model_notice()

    render_chat_history()

    user_message = st.chat_input("Type your feelings, situation, or dialogue here")
    if user_message:
        handle_user_message(user_message, selected_mode)


if __name__ == "__main__":
    main()
