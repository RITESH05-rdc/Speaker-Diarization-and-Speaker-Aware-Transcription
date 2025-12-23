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
from huggingface_hub import login

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

            # -------- LOAD & NORMALIZE AUDIO --------
            status.write("🎧 Normalizing audio...")
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            sf.write(audio_path, audio, sr)
            progress.progress(20)
            
            # -------- SPEAKER DIARIZATION --------
            status.write("🔍 Running speaker diarization...")
            annotation = diarization_pipeline({"audio": audio_path})
            progress.progress(40)


            st.subheader("📝 Speaker-wise Transcript")
            results = []
            # -------- BUILD RESULTS (EXACTLY LIKE COLAB) --------
            for segment, _, speaker in annotation.itertracks(yield_label=True):
                start = int(segment.start * sr)
                end = int(segment.end * sr)
            
                segment_audio = audio[start:end]
            
                # skip very short segments (< 0.5s)
                if len(segment_audio) < int(sr * 0.5):
                    continue
            
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_seg:
                    sf.write(tmp_seg.name, segment_audio, sr)
            
                transcription = whisper_model.transcribe(
                    tmp_seg.name,
                    fp16=False
                )
            
                os.remove(tmp_seg.name)
            
                results.append({
                    "speaker": speaker,
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": transcription["text"].strip()
                })
            
            progress.progress(80)

            import pandas as pd
            # -------- DISPLAY AS TABLE --------
            st.subheader("📝 Speaker-wise Transcript (Table View)")

            df = pd.DataFrame(results)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )


            progress.progress(100)
            status.success("✅ Processing complete!")

        st.balloons()

    if os.path.exists(audio_path):
        os.remove(audio_path)














