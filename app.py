# ================= IMPORTS =================
import streamlit as st
import tempfile
import librosa
import soundfile as sf
import whisper
from pyannote.audio import Pipeline
import random
import os

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
.main {
    color: #f5f5f5;
}
h1, h2, h3 {
    color: #00ffd5;
}
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    border: none;
    font-weight: bold;
}
.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #0072ff, #00c6ff);
}
.speaker-card {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.time {
    font-size: 0.85em;
    color: #bbbbbb;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("🎙️ Speaker Diarization & Transcription")
st.write(
    "Upload an audio file to **identify speakers** and generate "
    "**speaker-wise transcripts with timestamps**."
)

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    hf_token = st.secrets.get("HF_TOKEN", None)

    if hf_token is None:
        st.error("❌ Hugging Face token not found. Please add HF_TOKEN to Streamlit secrets.")
        st.stop()

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )

    whisper_model = whisper.load_model("tiny")
    return diarization_pipeline, whisper_model


# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader(
    "🎧 Upload Audio File",
    type=["wav", "mp3"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)
    st.divider()

    if st.button("🚀 Process Audio"):
        progress = st.progress(0)
        status = st.empty()

        with st.spinner("Analyzing speakers and transcribing..."):

            status.write("🔍 Running speaker diarization...")
            diarization = diarization_pipeline(audio_path)
            annotation = diarization.speaker_diarization
            progress.progress(30)

            status.write("🎧 Loading audio...")
            audio, sr = librosa.load(audio_path, sr=16000)
            progress.progress(50)

            st.subheader("📝 Speaker-wise Transcript")

            speaker_colors = {}
            palette = ["#00ffd5", "#ffb703", "#fb8500",
                       "#8ecae6", "#ff006e", "#8338ec"]

            for segment, _, speaker in annotation.itertracks(yield_label=True):

                if speaker not in speaker_colors:
                    speaker_colors[speaker] = random.choice(palette)

                start = int(segment.start * sr)
                end = int(segment.end * sr)
                segment_audio = audio[start:end]

                if len(segment_audio) < int(sr * 0.5):
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as seg:
                    sf.write(seg.name, segment_audio, sr)

                transcription = whisper_model.transcribe(
                    seg.name,
                    fp16=False
                )

                os.remove(seg.name)

                st.markdown(
                    f"""
                    <div class="speaker-card">
                        <h4 style="color:{speaker_colors[speaker]}">
                            🗣️ {speaker}
                        </h4>
                        <div class="time">
                            ⏱️ {segment.start:.1f}s → {segment.end:.1f}s
                        </div>
                        <p>{transcription["text"].strip()}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            progress.progress(100)
            status.success("✅ Processing complete!")

        st.balloons()

    if os.path.exists(audio_path):
        os.remove(audio_path)



