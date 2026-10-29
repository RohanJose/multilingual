# Indic Speech-to-Summary

Audio (Hindi / Malayalam / Tamil / Telugu) → Whisper large-v3 ASR → LLM summary, all via the [Groq](https://groq.com) API.
No local models are downloaded.

| Stage | Model (Groq) |
|---|---|
| ASR | `whisper-large-v3` (override: `GROQ_ASR_MODEL`) |
| Summary | `openai/gpt-oss-120b` (override: `GROQ_LLM_MODEL`) |

Set `GROQ_API_KEY` in `.env` (locally) or as a Space secret. Deployed as a Docker Space (see `Dockerfile`).

## Run
```
pip install -r requirements.txt
streamlit run app.py
pytest tests
```

## Evaluate (WER / CER / ROUGE / latency)
Put `clip.wav`, `clip.txt` (reference transcript), `clip.summary.txt` (reference summary) in a folder:
```
python evaluate.py sample_data --language Malayalam --out results.csv
```

