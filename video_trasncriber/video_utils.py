import os
import subprocess
from moviepy import VideoFileClip

class VideoUtils:
    def __init__(self, outputs_dir: str = "outputs"):
        self.outputs_dir = outputs_dir
        os.makedirs(self.outputs_dir, exist_ok=True)

    def extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video and save as 16-bit PCM WAV.
        Returns the path to outputs/audio.wav
        """
        audio_path = os.path.join(self.outputs_dir, "audio.wav")
        print(f"🎬 Extracting audio from {video_path} -> {audio_path}")
        clip = VideoFileClip(video_path)
        try:
            # Write WAV (PCM s16le) to ensure broad compatibility
            clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
        finally:
            clip.close()
        return audio_path

    def separate_vocals(self, audio_path: str, model: str = "htdemucs") -> str:
        """
        Use Demucs to separate vocals from the audio.
        Returns the path to the separated vocals.wav file.
        """
        output_dir = os.path.join(self.outputs_dir, "demucs")
        os.makedirs(output_dir, exist_ok=True)

        print(f"🎤 Separating vocals with Demucs ({model})...")

        # Run Demucs on CPU with a single job for stability on limited resources.
        # --two-stems=vocals outputs only vocals and accompaniment stems.
        cmd = [
            "demucs",
            "--two-stems=vocals",
            "-n", model,
            "--device", "cpu",
            "--jobs", "1",
            "-o", output_dir,
            audio_path,
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Demucs failed: {e}")
            raise

        # Demucs output structure: <output_dir>/<model>/<basename(audio_path)>/vocals.wav
        demucs_subdir = os.path.join(
            output_dir,
            model,
            os.path.splitext(os.path.basename(audio_path))[0]
        )
        vocals_path = os.path.join(demucs_subdir, "vocals.wav")

        if not os.path.exists(vocals_path):
            raise FileNotFoundError(f"Vocals not found at {vocals_path}")

        print(f"✅ Vocals saved at {vocals_path}")
        return vocals_path
