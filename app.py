# ================= IMPORTS =================
import streamlit as st
import tempfile
import librosa
import soundfile as sf
import whisper
from pyannote.audio import Pipeline
import os
import pandas as pd
from huggingface_hub import login

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Speaker Diarization & Speaker Aware Transcription",
    layout="wide"
)

# ================= SESSION STATE INIT =================
if "df" not in st.session_state:
    st.session_state.df = None

if "txt_output" not in st.session_state:
    st.session_state.txt_output = None

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "📋 Table View"

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
uploaded_file = st.file_uploader("🎧 Upload Audio File", type=["wav", "mp3"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)
    st.divider()

    # ================= PROCESS BUTTON =================
    if st.button("🚀 Process Audio") or st.session_state.df is not None:

        # ---------- RUN HEAVY PROCESSING ONLY ONCE ----------
        if st.session_state.df is None:

            progress = st.progress(0)
            status = st.empty()

            with st.spinner("Analyzing speakers and transcribing..."):

                audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                sf.write(audio_path, audio, sr)
                progress.progress(20)

                annotation = diarization_pipeline({"audio": audio_path})
                progress.progress(40)

                results = []

                for segment, _, speaker in annotation.itertracks(yield_label=True):
                    start = int(segment.start * sr)
                    end = int(segment.end * sr)
                    segment_audio = audio[start:end]

                    if len(segment_audio) < int(sr * 0.5):
                        continue

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as seg:
                        sf.write(seg.name, segment_audio, sr)

                    text = whisper_model.transcribe(seg.name, fp16=False)["text"].strip()
                    os.remove(seg.name)

                    results.append({
                        "speaker": speaker,
                        "start": round(segment.start, 2),
                        "end": round(segment.end, 2),
                        "text": text
                    })

                df = pd.DataFrame(results)

                if df.empty:
                    st.warning("⚠️ No valid speech segments detected.")
                    st.stop()

                # Store results permanently
                st.session_state.df = df

                # Build TXT output once
                txt = ""
                for _, r in df.iterrows():
                    txt += f"[{r['start']}s → {r['end']}s] {r['speaker']}: {r['text']}\n"

                st.session_state.txt_output = txt

                progress.progress(100)
                status.success("✅ Processing complete!")

        # ================= DISPLAY FROM SESSION STATE =================
        df = st.session_state.df

        

        # -------- VIEW TOGGLE (NO RESET) --------
        st.radio(
            "Choose View",
            ["📋 Table View", "💬 Chat View"],
            horizontal=True,
            key="view_mode"
        )

        # -------- TABLE VIEW --------
        if st.session_state.view_mode == "📋 Table View":
            st.dataframe(
                df[["start", "end", "speaker", "text"]],
                hide_index=True,
                use_container_width=True
            )

        # -------- CHAT VIEW --------
        else:
            for _, r in df.iterrows():
                st.markdown(
                    f"""
                    **{r['speaker']}**  
                    <small>{r['start']}s → {r['end']}s</small>  
                    {r['text']}
                    ---
                    """,
                    unsafe_allow_html=True
                )

        st.divider()

        # -------- TALK-TIME SUMMARY --------
        st.subheader("🧠 Talk-Time Summary")
        summary = (
            df.assign(Duration=df["end"] - df["start"])
              .groupby("speaker", as_index=False)["Duration"]
              .sum()
              .round(2)
        )
        st.dataframe(summary, hide_index=True, use_container_width=True)

        st.divider()
        
        # -------- DOWNLOAD (NO RESET) --------
        st.download_button(
            "⬇ Download as .txt",
            data=st.session_state.txt_output,
            file_name="speaker_transcript.txt",
            mime="text/plain",
            key="download_txt"
        )

    if os.path.exists(audio_path):
        os.remove(audio_path)



