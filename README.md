# Video Transcriber

Transcribe videos, separate vocals, and generate lyrics/subtitles. Supports:
- Downloading videos by URL
- Audio extraction
- Vocal separation (Demucs)
- Denoise + normalize
- Transcription (Whisper)
- Download cleaned audio and lyrics

## Quick start (local)

1) Create venv and install deps
- python -m venv .venv
- source .venv/bin/activate  (Windows: .venv\Scripts\activate)
- pip install -r requirements.txt

2) Run the app
- streamlit run main.py

3) Stop the app
- Ctrl+C in the terminal

## Deploy on Streamlit Community Cloud

Repo layout:
- main.py
- video_trasncriber/
  - downloader.py
  - lyric_extractor.py
  - video_utils.py
- requirements.txt
- packages.txt  (contains: ffmpeg)
- .streamlit/config.toml

Notes:
- CPU-only environment; Whisper and Demucs run on CPU.
- ffmpeg is installed via packages.txt.
- Public videos recommended for URL downloads.

## Usage

- Paste a YouTube URL and click “Download Video”, or upload a local file.
- Choose language (auto or specific).
- Optionally enable Demucs vocal separation and cleaned audio downloads.
- View and download lyrics for vocals and full audio.