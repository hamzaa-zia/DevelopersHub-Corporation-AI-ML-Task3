"""Streamlit interface for the mental health support chatbot demo.

Run with:
    streamlit run app.py

The interface uses the CPU-safe demo engine so reviewers can run the project
without a local GPU. Full Mistral 7B + LoRA inference remains in
`src/inference.py` for GPU or Colab use.
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


APP_TITLE = "Mental Health Support Chatbot"
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


def render_sidebar() -> None:
    """Show honest project limits and submission context."""
    with st.sidebar:
        st.title("Demo Mode")
        st.error(
            "This is a CPU-safe demonstration app. It does not load the real "
            "Mistral 7B LoRA adapter. Responses are selected using keyword "
            "rules only."
        )
        st.write(
            "The full fine-tuned Mistral 7B workflow is included in the "
            "training and inference scripts, but full inference requires a "
            "CUDA GPU or Colab GPU runtime."
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


def render_chat_history() -> None:
    """Display every previous user and assistant message as chat bubbles."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def handle_user_message(user_message: str) -> None:
    """Save the user message, generate a response, and save the assistant reply."""
    st.session_state.messages.append({"role": "user", "content": user_message})

    with st.chat_message("user"):
        st.write(user_message)

    assistant_response = generate_demo_response(user_message)
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
    st.caption("CPU-safe submission demo for the fine-tuned chatbot workflow")

    render_sidebar()
    render_demo_notice()
    render_chat_history()

    user_message = st.chat_input("Type your feelings, situation, or dialogue here")
    if user_message:
        handle_user_message(user_message)


if __name__ == "__main__":
    main()
