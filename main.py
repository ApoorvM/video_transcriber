# Thread/env limits first to reduce resource pressure and avoid sudden kills
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # macOS fallback if MPS is flaky

import gc
import time
import torch
torch.set_num_threads(1)

import streamlit as st
from video_trasncriber.lyric_extractor import LyricExtractor
from video_trasncriber.downloader import download_video, DownloadError

st.set_page_config(page_title="Video Lyric Extractor", page_icon="🎵", layout="centered")
st.title("Video Lyric Extractor")

# 1) Download by URL
st.subheader("Download a video from URL")
video_url = st.text_input("Paste a YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")

downloaded_path_placeholder = st.empty()
downloaded_path = st.session_state.get("downloaded_path", None)

col_dl1, col_dl2 = st.columns([1, 1])
with col_dl1:
    if st.button("Download Video"):
        if not video_url.strip():
            st.warning("Please paste a valid YouTube URL.")
        else:
            with st.spinner("Downloading video..."):
                try:
                    saved_path = download_video(video_url, output_dir="downloads")
                    st.session_state["downloaded_path"] = saved_path
                    downloaded_path = saved_path
                    downloaded_path_placeholder.success(f"Downloaded to: {saved_path}")
                except DownloadError as e:
                    downloaded_path_placeholder.error(f"Download failed: {e}")
                except Exception as e:
                    downloaded_path_placeholder.error(f"Unexpected error: {e}")

with col_dl2:
    if st.button("Clear Download"):
        if "downloaded_path" in st.session_state:
            st.session_state.pop("downloaded_path")
        downloaded_path = None
        downloaded_path_placeholder.info("Cleared downloaded video selection.")

# 2) Or upload a local video
st.subheader("Or upload a local video")
uploaded_file = st.file_uploader("Upload your video file", type=["mp4", "mov"])

# Whisper language selection (auto = autodetect)
language = st.selectbox(
    "Select song language (improves transcription accuracy)",
    options=[
        "auto",  # autodetect
        "hi",    # Hindi
        "en",    # English
        "bn",    # Bengali
        "ta",    # Tamil
        "te",    # Telugu
        "pa",    # Punjabi
        "mr",    # Marathi
        "gu",    # Gujarati
        "ur",    # Urdu
    ],
    index=0
)

# Toggles
enable_audio_downloads = st.checkbox(
    "Make cleaned audio files available for download (vocals_norm.wav, audio_norm.wav)",
    value=True
)
use_demucs = st.checkbox("Separate vocals (Demucs)", value=True)

step_labels = [
    "Audio Extraction",
    "Vocal Separation",
    "Denoise & Normalize",
    "Model Load",
    "Transcribe Vocals",
    "Transcribe Audio",
    "Save Files",
]
# Estimated durations (seconds) — tune to your machine
step_estimates = [5, 25, 5, 8, 30, 30, 2]

@st.cache_resource(show_spinner=False)
def load_whisper_model(name="small", device="cpu"):
    import whisper
    return whisper.load_model(name, device=device)

def process_video(video_path, outputs_dir="outputs", lang="auto"):
    extractor = LyricExtractor(outputs_dir=outputs_dir)
    bars = [st.progress(0, text=f"{label} Progress") for label in step_labels]

    # 1) Audio extraction
    idx = 0
    bars[idx].progress(10, text=f"{step_labels[idx]}: Starting")
    audio_path = extractor.silent_call(extractor.vu.extract_audio, video_path)
    for i in (25, 50, 75, 100):
        bars[idx].progress(i, text=f"{step_labels[idx]}: {i}%")
        time.sleep(step_estimates[idx] / 4)
    bars[idx].progress(100, text=f"{step_labels[idx]}: Done")
    gc.collect()

    # 2) Demucs separation (optional)
    idx = 1
    bars[idx].progress(10, text=f"{step_labels[idx]}: Starting")
    if use_demucs:
        vocals_path = extractor.silent_call(extractor.vu.separate_vocals, audio_path, "htdemucs")
    else:
        vocals_path = audio_path
    for i in (25, 50, 75, 100):
        bars[idx].progress(i, text=f"{step_labels[idx]}: {i}%")
        time.sleep(step_estimates[idx] / 4)
    bars[idx].progress(100, text=f"{step_labels[idx]}: Done")
    gc.collect()

    # 3) Denoise & Normalize
    idx = 2
    bars[idx].progress(10, text=f"{step_labels[idx]}: Starting")
    norm_audio_path = os.path.join(outputs_dir, "audio_norm.wav")
    norm_vocals_path = os.path.join(outputs_dir, "vocals_norm.wav")
    den_audio_path = os.path.join(outputs_dir, "audio_denoise.wav")
    den_vocal_path = os.path.join(outputs_dir, "vocals_denoise.wav")

    extractor.denoise_audio(audio_path, den_audio_path)
    extractor.denoise_audio(vocals_path, den_vocal_path)
    extractor.normalize_audio(den_audio_path, norm_audio_path)
    extractor.normalize_audio(den_vocal_path, norm_vocals_path)
    for i in (25, 50, 75, 100):
        bars[idx].progress(i, text=f"{step_labels[idx]}: {i}%")
        time.sleep(step_estimates[idx] / 4)
    bars[idx].progress(100, text=f"{step_labels[idx]}: Done")
    gc.collect()

    # 4) Whisper model load (cached, small+CPU for stability)
    idx = 3
    bars[idx].progress(10, text=f"{step_labels[idx]}: Starting")
    model = load_whisper_model(name="small", device="cpu")
    for i in (25, 50, 75, 100):
        bars[idx].progress(i, text=f"{step_labels[idx]}: {i}%")
        time.sleep(step_estimates[idx] / 4)
    bars[idx].progress(100, text=f"{step_labels[idx]}: Done")
    gc.collect()

    # Build transcription kwargs
    transcribe_kwargs = {}
    if lang and lang != "auto":
        transcribe_kwargs["language"] = lang

    # 5) Transcribe Vocals
    idx = 4
    bars[idx].progress(10, text=f"{step_labels[idx]}: Starting")
    vocals_result = extractor.silent_call(
        model.transcribe, norm_vocals_path, fp16=False, **transcribe_kwargs
    )
    vocals_lyrics = vocals_result.get("text", "")
    for i in (25, 50, 75, 100):
        bars[idx].progress(i, text=f"{step_labels[idx]}: {i}%")
        time.sleep(step_estimates[idx] / 4)
    bars[idx].progress(100, text=f"{step_labels[idx]}: Done")
    gc.collect()

    # 6) Transcribe Audio
    idx = 5
    bars[idx].progress(10, text=f"{step_labels[idx]}: Starting")
    audio_result = extractor.silent_call(
        model.transcribe, norm_audio_path, fp16=False, **transcribe_kwargs
    )
    audio_lyrics = audio_result.get("text", "")
    for i in (25, 50, 75, 100):
        bars[idx].progress(i, text=f"{step_labels[idx]}: {i}%")
        time.sleep(step_estimates[idx] / 4)
    bars[idx].progress(100, text=f"{step_labels[idx]}: Done")
    gc.collect()

    # 7) Save lyrics files
    idx = 6
    bars[idx].progress(10, text=f"{step_labels[idx]}: Starting")
    vocals_file = os.path.join(outputs_dir, "lyrics_from_vocals.txt")
    audio_file = os.path.join(outputs_dir, "lyrics_from_audio.txt")
    with open(vocals_file, "w", encoding="utf-8") as f:
        f.write(vocals_lyrics)
    with open(audio_file, "w", encoding="utf-8") as f:
        f.write(audio_lyrics)
    for i in (25, 50, 75, 100):
        bars[idx].progress(i, text=f"{step_labels[idx]}: {i}%")
        time.sleep(step_estimates[idx] / 4)
    bars[idx].progress(100, text=f"{step_labels[idx]}: Done")
    gc.collect()

    return {
        "vocals_lyrics": vocals_lyrics,
        "audio_lyrics": audio_lyrics,
        "vocals_file": vocals_file,
        "audio_file": audio_file,
        "norm_vocals_path": norm_vocals_path,
        "norm_audio_path": norm_audio_path,
    }

# Decide which video to process
source_video_path = None

# Prefer downloaded path if available
if downloaded_path:
    source_video_path = downloaded_path

# Fallback to uploaded file
if uploaded_file is not None and source_video_path is None:
    source_video_path = "uploaded_video.mp4"
    with open(source_video_path, "wb") as f:
        f.write(uploaded_file.read())

# Run pipeline
if source_video_path:
    st.info("Processing each step. This may take several minutes for long videos.")
    try:
        result = process_video(source_video_path, lang=language)
    except Exception as e:
        st.error(f"Processing failed: {e}")
        raise

    st.header("Lyrics from Vocals")
    st.text_area("Vocals Lyrics", value=result["vocals_lyrics"], height=300)

    st.header("Lyrics from Original Audio")
    st.text_area("Original Audio Lyrics", value=result["audio_lyrics"], height=300)

    # Download lyrics
    st.download_button(
        "Download Vocals Lyrics",
        open(result["vocals_file"], "rb").read(),
        file_name="lyrics_from_vocals.txt"
    )
    st.download_button(
        "Download Audio Lyrics",
        open(result["audio_file"], "rb").read(),
        file_name="lyrics_from_audio.txt"
    )

    # Optional: download cleaned audio
    if enable_audio_downloads:
        st.subheader("Download Cleaned Audio")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download vocals_norm.wav",
                open(result["norm_vocals_path"], "rb").read(),
                file_name="vocals_norm.wav",
                mime="audio/wav"
            )
        with col2:
            st.download_button(
                "Download audio_norm.wav",
                open(result["norm_audio_path"], "rb").read(),
                file_name="audio_norm.wav",
                mime="audio/wav"
            )
