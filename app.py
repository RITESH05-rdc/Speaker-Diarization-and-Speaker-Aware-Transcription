# ================= IMPORTS =================
import os
os.environ["PYANNOTE_AUDIO_DISABLE_TELEMETRY"] = "1"

import streamlit as st
import tempfile
import librosa
import soundfile as sf
import whisper
from pyannote.audio import Pipeline
import random

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Speaker Diarization & Transcription",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.main { color: #f5f5f5; }
h1, h2, h3 { color: #00ffd5; }
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    border: none;
    font-weight: bold;
}
.speaker-card {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 12px;
}
.time {
    font-size: 0.85em;
    color: #bbbbbb;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("🎙️ Speaker Diarization & Transcription")
st.write("Upload an audio file to identify speakers and generate transcripts.")

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    hf_token = st.secrets.get("HF_TOKEN")
    if not hf_token:
        st.error("HF_TOKEN missing in Streamlit secrets.")
        st.stop()

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )

    whisper_model = whisper.load_model("tiny")
    return diarization_pipeline, whisper_model


diarization_pipeline, whisper_model = load_models()

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("🎧 Upload Audio File", type=["wav", "mp3"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)

    if st.button("🚀 Process Audio"):
        with st.spinner("Processing..."):

            # ---- DIARIZATION ----
            diarization = diarization_pipeline({
                "audio": audio_path,
                "sample_rate": 16000
            })

            audio, sr = librosa.load(audio_path, sr=16000)
            annotation = diarization.speaker_diarization

            st.subheader("📝 Speaker-wise Transcript")

            colors = {}
            palette = ["#00ffd5", "#ffb703", "#fb8500", "#8ecae6"]

            for segment, _, speaker in annotation.itertracks(yield_label=True):

                colors.setdefault(speaker, random.choice(palette))

                start = int(segment.start * sr)
                end = int(segment.end * sr)
                seg_audio = audio[start:end]

                if len(seg_audio) < sr * 0.5:
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as seg:
                    sf.write(seg.name, seg_audio, sr)
                    text = whisper_model.transcribe(seg.name, fp16=False)["text"]
                    os.remove(seg.name)

                st.markdown(
                    f"""
                    <div class="speaker-card">
                        <h4 style="color:{colors[speaker]}">🗣 {speaker}</h4>
                        <div class="time">{segment.start:.1f}s → {segment.end:.1f}s</div>
                        <p>{text.strip()}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.success("✅ Done!")
