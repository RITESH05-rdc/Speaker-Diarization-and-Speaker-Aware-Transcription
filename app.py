# ================= IMPORTS =================
import streamlit as st
import tempfile
import librosa
import soundfile as sf
import whisper
from pyannote.audio import Pipeline
from huggingface_hub import login
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
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
}
.time {
    font-size: 0.85em;
    color: #bbbbbb;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("🎙️ Speaker Diarization & Transcription")
st.write(
    "Upload an audio file to **identify speakers** and generate "
    "**clean, speaker-wise transcripts**."
)

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    hf_token = st.secrets.get("HF_TOKEN")

    if not hf_token:
        st.error("❌ Hugging Face token not found.")
        st.stop()

    login(token=hf_token)

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1"
    )

    whisper_model = whisper.load_model("base")
    return diarization_pipeline, whisper_model


diarization_pipeline, whisper_model = load_models()

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

            # -------- NORMALIZE AUDIO --------
            status.write("🎧 Normalizing audio...")
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            sf.write(audio_path, audio, sr)
            progress.progress(20)

            # -------- SPEAKER DIARIZATION --------
            status.write("🔍 Running speaker diarization...")
            annotation = diarization_pipeline({"audio": audio_path})
            progress.progress(40)

            # -------- LOAD AUDIO FOR SLICING --------
            audio, sr = librosa.load(audio_path, sr=16000)
            progress.progress(50)

            st.subheader("📝 Speaker-wise Transcript")

            # -------- STORAGE --------
            speaker_segments = {}
            speaker_times = {}

            # -------- TRANSCRIBE SEGMENTS --------
            for segment, _, speaker in annotation.itertracks(yield_label=True):

                start_sample = int(segment.start * sr)
                end_sample = int(segment.end * sr)
                segment_audio = audio[start_sample:end_sample]

                # Skip very short segments
                if len(segment_audio) < int(sr * 0.5):
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as seg:
                    sf.write(seg.name, segment_audio, sr)

                result = whisper_model.transcribe(
                    seg.name,
                    fp16=False
                )

                os.remove(seg.name)

                text = result["text"].strip()
                if not text:
                    continue

                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                    speaker_times[speaker] = [segment.start, segment.end]
                else:
                    speaker_times[speaker][1] = segment.end

                speaker_segments[speaker].append(text)

            progress.progress(80)

            # -------- DISPLAY MERGED CARDS --------
            palette = ["#00ffd5", "#ffb703", "#fb8500",
                       "#8ecae6", "#ff006e", "#8338ec"]
            speaker_colors = {}

            for speaker, texts in speaker_segments.items():
                if speaker not in speaker_colors:
                    speaker_colors[speaker] = random.choice(palette)

                start_t, end_t = speaker_times[speaker]
                full_text = " ".join(texts)

                st.markdown(
                    f"""
                    <div class="speaker-card">
                        <h4 style="color:{speaker_colors[speaker]}">
                            🗣️ {speaker}
                        </h4>
                        <div class="time">
                            ⏱️ {start_t:.1f}s → {end_t:.1f}s
                        </div>
                        <p>{full_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            progress.progress(100)
            status.success("✅ Processing complete!")
            st.balloons()

    if os.path.exists(audio_path):
        os.remove(audio_path)
