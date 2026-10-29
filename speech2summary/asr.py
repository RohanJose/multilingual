"""Speech recognition with Whisper on Groq."""
import io

import librosa
import numpy as np
import soundfile as sf

from . import groq_client
from .config import ASR_MODEL, LANGUAGES, SAMPLE_RATE


def load_audio(path: str) -> np.ndarray:
    """Load any supported audio file as 16 kHz mono float32."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio


def transcribe(audio_path: str, language: str) -> str:
    audio = load_audio(audio_path)
    if audio.size == 0:
        return ""
    # Re-encode as FLAC so long recordings stay under Groq's upload size limit.
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="FLAC")
    return groq_client.transcribe(buf.getvalue(), "audio.flac", ASR_MODEL, LANGUAGES[language].code).strip()
