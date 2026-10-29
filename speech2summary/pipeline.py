import time
from dataclasses import dataclass

from . import asr, summarize as summ


@dataclass
class Result:
    transcript: str
    summary: str
    method: str
    asr_seconds: float
    summary_seconds: float


def speech_to_summary(audio_path: str, language: str) -> Result:
    t0 = time.perf_counter()
    transcript = asr.transcribe(audio_path, language)
    t1 = time.perf_counter()
    summary, method = summ.summarize(transcript, language)
    t2 = time.perf_counter()
    return Result(transcript, summary, method, t1 - t0, t2 - t1)
