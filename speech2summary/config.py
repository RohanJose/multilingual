"""Language registry and Groq model settings.

Models can be overridden with env vars: GROQ_ASR_MODEL, GROQ_LLM_MODEL.
"""
import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path=Path(__file__).resolve().parent.parent / ".env"):
    """Minimal .env loader so API keys never have to live in source code."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

SAMPLE_RATE = 16_000

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
ASR_MODEL = os.environ.get("GROQ_ASR_MODEL", "whisper-large-v3")
LLM_MODEL = os.environ.get("GROQ_LLM_MODEL", "openai/gpt-oss-120b")


@dataclass(frozen=True)
class Language:
    name: str
    code: str  # ISO 639-1, passed to Whisper


_LANGS = [
    Language("Hindi", "hi"),
    Language("Malayalam", "ml"),
    Language("Tamil", "ta"),
    Language("Telugu", "te"),
]

LANGUAGES = {l.name: l for l in _LANGS}
