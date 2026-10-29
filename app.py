import os
import tempfile

import streamlit as st

from speech2summary import speech_to_summary
from speech2summary.config import LANGUAGES

st.set_page_config(page_title="Indic Speech-to-Summary", page_icon="🎙️", layout="centered")
st.title("🎙️ Indic Speech-to-Summary")
st.caption("Whisper ASR + LLM summarization, powered by Groq")

language = st.selectbox("Language", list(LANGUAGES))
audio_file = st.file_uploader("Upload a recording", type=["wav", "mp3", "flac", "m4a", "ogg"])

if audio_file:
    st.audio(audio_file)
    if st.button("Generate Summary", type="primary"):
        suffix = os.path.splitext(audio_file.name)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_file.getvalue())
        try:
            with st.spinner("Transcribing and summarizing …"):
                st.session_state["result"] = speech_to_summary(tmp.name, language)
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"Processing failed: {e}")
        finally:
            os.unlink(tmp.name)

res = st.session_state.get("result")
if res:
    st.subheader("Transcript")
    st.text_area("transcript", res.transcript, height=200, label_visibility="collapsed")
    st.download_button("Download transcript", res.transcript, "transcript.txt")
    st.subheader("Summary")
    st.text_area("summary", res.summary, height=140, label_visibility="collapsed")
    st.download_button("Download summary", res.summary, "summary.txt")
    st.caption(f"ASR {res.asr_seconds:.1f}s · Summary {res.summary_seconds:.1f}s · {res.method}")
