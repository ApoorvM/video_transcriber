import os
import time
import contextlib
import gc
import numpy as np
import soundfile as sf

from video_trasncriber.video_utils import VideoUtils

try:
    import noisereduce as nr
    USE_DENOISE = True
except ImportError:
    USE_DENOISE = False


class LyricExtractor:
    def __init__(self, outputs_dir="outputs"):
        self.outputs_dir = outputs_dir
        os.makedirs(self.outputs_dir, exist_ok=True)
        self.vu = VideoUtils(outputs_dir=outputs_dir)

    @staticmethod
    def log_step(step_name, start_time):
        elapsed = time.time() - start_time
        print(f"⏱️ {step_name} took {elapsed:.2f} seconds.\n")

    @staticmethod
    def normalize_audio(input_path, output_path):
        data, samplerate = sf.read(input_path)
        peak = np.max(np.abs(data)) if data.size > 0 else 0.0
        target_peak = 0.1  # approx -20 dBFS
        if peak > 0:
            data = data * (target_peak / peak)
        sf.write(output_path, data, samplerate)

    @staticmethod
    def denoise_audio(input_path, output_path):
        data, samplerate = sf.read(input_path)
        if not USE_DENOISE:
            sf.write(output_path, data, samplerate)
            return
        reduced_noise = nr.reduce_noise(y=data, sr=samplerate)
        sf.write(output_path, reduced_noise, samplerate)

    @staticmethod
    def to_mono(in_path, out_path):
        """Optional: downmix to mono to reduce memory footprint."""
        data, sr = sf.read(in_path)
        if hasattr(data, "ndim") and data.ndim > 1:
            data = np.mean(data, axis=1)
        sf.write(out_path, data, sr)

    @staticmethod
    def silent_call(func, *args, **kwargs):
        with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(open(os.devnull, "w")):
            return func(*args, **kwargs)

    def run_pipeline(self, video_path: str, whisper_model, language: str = "auto"):
        """
        Run the full pipeline:
        - Extract audio
        - Separate vocals (Demucs)
        - Denoise + Normalize (both tracks)
        - Transcribe using provided Whisper model
        - Save lyrics to text files

        Args:
            video_path: path to input video
            whisper_model: a loaded Whisper model instance (already loaded outside)
            language: language code for Whisper; use "auto" for autodetect

        Returns:
            dict with paths and transcription texts
        """
        # Section 1: Extract audio
        print("=== Starting Audio Extraction ===")
        t0 = time.time()
        audio_path = self.silent_call(self.vu.extract_audio, video_path)
        self.log_step("Audio extraction", t0)
        gc.collect()

        # Optional: mono downmix before Demucs (uncomment to reduce memory)
        # mono_audio_path = os.path.join(self.outputs_dir, "audio_mono.wav")
        # self.to_mono(audio_path, mono_audio_path)
        # demucs_input = mono_audio_path
        demucs_input = audio_path

        # Section 2: Vocal Separation
        print("=== Starting Vocal Separation (Demucs) ===")
        t0 = time.time()
        vocals_path = self.silent_call(self.vu.separate_vocals, demucs_input, "htdemucs")
        self.log_step("Vocal separation (Demucs)", t0)
        gc.collect()

        # Section 3: Denoising & Normalization
        print("=== Starting Denoising & Normalization ===")
        t0 = time.time()
        norm_audio_path = os.path.join(self.outputs_dir, "audio_norm.wav")
        norm_vocals_path = os.path.join(self.outputs_dir, "vocals_norm.wav")
        den_audio_path = os.path.join(self.outputs_dir, "audio_denoise.wav")
        den_vocal_path = os.path.join(self.outputs_dir, "vocals_denoise.wav")

        self.denoise_audio(audio_path, den_audio_path)
        self.denoise_audio(vocals_path, den_vocal_path)
        self.normalize_audio(den_audio_path, norm_audio_path)
        self.normalize_audio(den_vocal_path, norm_vocals_path)
        self.log_step("Denoising & normalization", t0)
        gc.collect()

        # Section 4/5: Transcribe
        transcribe_kwargs = {}
        if language and language != "auto":
            transcribe_kwargs["language"] = language

        print("=== Transcribing Clean Vocals ===")
        t0 = time.time()
        vocals_result = self.silent_call(
            whisper_model.transcribe, norm_vocals_path, fp16=False, **transcribe_kwargs
        )
        vocals_lyrics = vocals_result.get("text", "")
        self.log_step("Transcription (vocals)", t0)
        gc.collect()

        print("=== Transcribing Clean Original Audio ===")
        t0 = time.time()
        audio_result = self.silent_call(
            whisper_model.transcribe, norm_audio_path, fp16=False, **transcribe_kwargs
        )
        audio_lyrics = audio_result.get("text", "")
        self.log_step("Transcription (original audio)", t0)
        gc.collect()

        # Section 6: Save text files
        print("=== Saving Lyrics to Files ===")
        t0 = time.time()
        vocals_file = os.path.join(self.outputs_dir, "lyrics_from_vocals.txt")
        audio_file = os.path.join(self.outputs_dir, "lyrics_from_audio.txt")
        with open(vocals_file, "w", encoding="utf-8") as f:
            f.write(vocals_lyrics)
        with open(audio_file, "w", encoding="utf-8") as f:
            f.write(audio_lyrics)
        self.log_step("Save transcripts to file", t0)

        return {
            "audio_path": audio_path,
            "vocals_path": vocals_path,
            "norm_audio_path": norm_audio_path,
            "norm_vocals_path": norm_vocals_path,
            "vocals_lyrics": vocals_lyrics,
            "audio_lyrics": audio_lyrics,
            "vocals_file": vocals_file,
            "audio_file": audio_file,
        }
