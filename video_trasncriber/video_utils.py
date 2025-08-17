import os
import subprocess
from moviepy import VideoFileClip

class VideoUtils:
    def __init__(self, outputs_dir="outputs"):
        self.outputs_dir = outputs_dir
        os.makedirs(self.outputs_dir, exist_ok=True)

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video and save as 16-bit PCM WAV."""
        audio_path = os.path.join(self.outputs_dir, "audio.wav")
        print(f"🎬 Extracting audio from {video_path} -> {audio_path}")
        clip = VideoFileClip(video_path)
        # Write WAV with 16-bit PCM (broadly compatible)
        clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
        clip.close()
        return audio_path

    def separate_vocals(self, audio_path: str, model: str = "htdemucs") -> str:
        """Use Demucs to separate vocals from audio."""
        output_dir = os.path.join(self.outputs_dir, "demucs")
        os.makedirs(output_dir, exist_ok=True)
        print(f"🎤 Separating vocals with Demucs ({model})...")

        try:
            # Force CPU, single worker to reduce memory/VRAM pressure; silence output
            subprocess.run(
                [
                    "demucs",
                    "--two-stems=vocals",
                    "-n", model,
                    "--jobs", "1",        # single worker for stability
                    "--device", "cpu",    # avoid GPU VRAM spikes / MPS issues
                    "-o", output_dir,
                    audio_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Demucs failed: {e}")
            raise

        # Locate vocals.wav in Demucs output
        demucs_subdir = os.path.join(
            output_dir, model, os.path.splitext(os.path.basename(audio_path))[0]
        )
        vocals_path = os.path.join(demucs_subdir, "vocals.wav")

        if not os.path.exists(vocals_path):
            raise FileNotFoundError(f"Vocals not found at {vocals_path}")

        print(f"✅ Vocals saved at {vocals_path}")
        return vocals_path
