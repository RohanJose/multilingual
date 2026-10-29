"""Thin Groq REST client (OpenAI-compatible); no extra SDK needed."""
import os
import time

import requests

from .config import GROQ_BASE_URL


def _post(path: str, retries: int = 3, **kwargs) -> dict:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set (put it in .env)")
    for attempt in range(retries):
        r = requests.post(f"{GROQ_BASE_URL}{path}", headers={"Authorization": f"Bearer {key}"},
                          timeout=120, **kwargs)
        if r.status_code == 429 and attempt < retries - 1:
            time.sleep(float(r.headers.get("retry-after", 2 ** attempt)))
            continue
        if not r.ok:
            raise RuntimeError(f"Groq API error {r.status_code}: {r.text[:300]}")
        return r.json()


def transcribe(audio_bytes: bytes, filename: str, model: str, language: str) -> str:
    data = _post("/audio/transcriptions", files={"file": (filename, audio_bytes)},
                 data={"model": model, "language": language, "temperature": "0",
                       "response_format": "json"})
    return data["text"]


def chat(messages: list[dict], model: str, max_tokens: int = 2000) -> str:
    data = _post("/chat/completions",
                 json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens})
    return data["choices"][0]["message"]["content"].strip()
