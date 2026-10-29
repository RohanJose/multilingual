"""Summarization with a Groq-hosted LLM; the summary is written in the speaker's language."""
from . import groq_client
from .config import LANGUAGES, LLM_MODEL


def summarize(text: str, language: str) -> tuple[str, str]:
    """Return (summary_in_source_language, method_description)."""
    text = text.strip()
    if not text:
        return "", "no input"
    summary = groq_client.chat(
        [
            {"role": "system", "content": (
                f"You summarize speech transcripts. Write a concise summary (2-4 sentences) in {language}, "
                "using only information in the transcript. Output only the summary.")},
            {"role": "user", "content": text},
        ],
        LLM_MODEL,
    )
    return summary, f"Groq {LLM_MODEL}"
