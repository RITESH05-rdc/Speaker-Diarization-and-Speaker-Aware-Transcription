import streamlit as st
import tempfile
import librosa
import soundfile as sf
import whisper
from pyannote.audio import Pipeline
import random

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Speaker Diarization & Transcription",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
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

# ---------------- TITLE ----------------
st.title("🎙️ Speaker Diarization & Transcription")
st.write(
    "Upload an audio file to **identify speakers** and get a "
    "**speaker-wise transcript** with timestamps."
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=st.secrets["HF_TOKEN"]
    )
    whisper_model = whisper.load_model("base")
    return diarization_pipeline, whisper_model

pipeline, whisper_model = load_models()

# ---------------- FILE UPLOAD ----------------
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
            diarization = pipeline(audio_path)
            progress.progress(30)

            status.write("🎧 Loading audio...")
            audio, sr = librosa.load(audio_path, sr=16000)
            progress.progress(50)

            st.subheader("📝 Speaker-wise Transcript")

            speaker_colors = {}
            color_palette = [
                "#00ffd5", "#ffb703", "#fb8500",
                "#8ecae6", "#ff006e", "#8338ec"
            ]

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_colors:
                    speaker_colors[speaker] = random.choice(color_palette)

                start = int(turn.start * sr)
                end = int(turn.end * sr)
                segment = audio[start:end]

                if len(segment) < sr * 0.5:
                    continue

                sf.write("temp.wav", segment, sr)
                text = whisper_model.transcribe(
                    "temp.wav",
                    fp16=False
                )["text"]

                st.markdown(
                    f"""
                    <div class="speaker-card">
                        <h4 style="color:{speaker_colors[speaker]}">
                            🗣️ {speaker}
                        </h4>
                        <div class="time">
                            ⏱️ {turn.start:.1f}s → {turn.end:.1f}s
                        </div>
                        <p>{text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            progress.progress(100)
            status.success("✅ Processing complete!")

        st.balloons()
