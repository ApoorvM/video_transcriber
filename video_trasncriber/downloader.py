import os
import uuid
import subprocess
import shutil
from typing import Optional

# Prefer yt-dlp (most reliable, supports many sites)
_YTDLP_OK = False
try:
    import yt_dlp  # type: ignore
    _YTDLP_OK = True
except Exception:
    _YTDLP_OK = False

# Fallback: pytube (YouTube only)
_PYTUBE_OK = False
try:
    from pytube import YouTube  # type: ignore
    from pytube.exceptions import PytubeError, RegexMatchError, VideoUnavailable  # type: ignore
    _PYTUBE_OK = True
except Exception:
    _PYTUBE_OK = False


class DownloadError(Exception):
    pass


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def _download_with_ytdlp(url: str, output_dir: str, filename: Optional[str]) -> str:
    """
    Download using yt-dlp, selecting best video+audio and merging to MP4.
    Returns the final file path (preferably .mp4).
    """
    if not _ffmpeg_exists():
        raise DownloadError(
            "ffmpeg not found. Please install ffmpeg and ensure it is on PATH."
        )

    _ensure_dir(output_dir)
    base = filename or f"{uuid.uuid4().hex}"
    out_template = os.path.join(output_dir, base + ".%(ext)s")

    ydl_opts = {
        "outtmpl": out_template,
        "format": "bv*+ba/b",             # best video + best audio; fallback to best
        "merge_output_format": "mp4",     # final merged container
        "retries": 3,
        "nocheckcertificate": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded = ydl.prepare_filename(info)  # pre-merge filename
    except Exception as e:
        raise DownloadError(f"yt-dlp failed: {e}")

    # Resolve to final .mp4 if available
    root, ext = os.path.splitext(downloaded)
    target_mp4 = root + ".mp4"
    if os.path.exists(target_mp4):
        return target_mp4
    if ext.lower() == ".mp4" and os.path.exists(downloaded):
        return downloaded

    guess_mp4 = os.path.join(output_dir, base + ".mp4")
    if os.path.exists(guess_mp4):
        return guess_mp4

    if os.path.exists(downloaded):
        return downloaded

    raise DownloadError("yt-dlp reported success but final file was not found.")


def _merge_av_ffmpeg(video_path: str, audio_path: str, out_path: str) -> None:
    """
    Merge adaptive video+audio into MP4. Try stream copy; if audio codec incompatible,
    transcode audio to AAC to ensure MP4 compatibility.
    """
    if not _ffmpeg_exists():
        raise DownloadError(
            "ffmpeg is required to merge adaptive video/audio. Install ffmpeg and re-try."
        )

    # First try a fast container copy
    cmd_copy = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c", "copy",
        out_path
    ]
    result = subprocess.run(cmd_copy, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode == 0 and os.path.exists(out_path):
        return

    # Fallback: copy video, transcode audio to AAC for MP4 compatibility
    cmd_aac = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        out_path
    ]
    result2 = subprocess.run(cmd_aac, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result2.returncode != 0 or not os.path.exists(out_path):
        raise DownloadError("ffmpeg failed to merge video+audio into MP4.")


def _download_with_pytube(url: str, output_dir: str, filename: Optional[str]) -> str:
    """
    Pytube fallback:
      1) Try highest progressive MP4 (video+audio in one file)
      2) Otherwise: best adaptive video MP4 + best audio -> merge via ffmpeg
    Returns the final mp4 path.
    """
    _ensure_dir(output_dir)

    try:
        yt = YouTube(url)
    except Exception as e:
        raise DownloadError(f"pytube init failed: {e}")

    target_filename = filename or f"{yt.video_id}.mp4"
    out_path = os.path.join(output_dir, target_filename)

    # Try progressive first (contains both audio+video)
    try:
        prog = (
            yt.streams
            .filter(progressive=True, file_extension="mp4")
            .order_by("resolution").desc()
            .first()
        )
        if prog:
            prog.download(output_path=output_dir, filename=target_filename)
            if os.path.exists(out_path):
                return out_path
    except (PytubeError, RegexMatchError, VideoUnavailable):
        # Continue to adaptive fallback
        pass

    # Adaptive fallback (separate video + audio -> merge)
    try:
        best_video = (
            yt.streams
            .filter(only_video=True, file_extension="mp4")
            .order_by("resolution").desc()
            .first()
        )
        best_audio = (
            yt.streams
            .filter(only_audio=True)
            .order_by("abr").desc()
            .first()
        )
        if best_video is None or best_audio is None:
            raise DownloadError("No suitable adaptive streams found (video or audio missing).")

        tmp_video = os.path.join(output_dir, f"{yt.video_id}_video.mp4")
        # Audio may be webm/others; we'll still write .m4a/.mp4; ffmpeg handles inputs.
        tmp_audio = os.path.join(output_dir, f"{yt.video_id}_audio.m4a")

        best_video.download(output_path=output_dir, filename=os.path.basename(tmp_video))
        best_audio.download(output_path=output_dir, filename=os.path.basename(tmp_audio))

        _merge_av_ffmpeg(tmp_video, tmp_audio, out_path)

        # Cleanup temp files
        for p in (tmp_video, tmp_audio):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

        if os.path.exists(out_path):
            return out_path

        raise DownloadError("Adaptive merge completed but final file not found.")
    except (PytubeError, RegexMatchError, VideoUnavailable, subprocess.CalledProcessError) as e:
        raise DownloadError(f"pytube adaptive failed: {e}")


def download_video(url: str, output_dir: str = "downloads", filename: Optional[str] = None) -> str:
    """
    Reliable downloader:
      - Use yt-dlp if available (handles many sites; merges A/V into MP4).
      - Otherwise fallback to pytube for YouTube.
    Returns path to downloaded video file (contains both video and audio).
    Raises DownloadError on failure.
    """
    if _YTDLP_OK:
        return _download_with_ytdlp(url, output_dir, filename)

    if _PYTUBE_OK:
        return _download_with_pytube(url, output_dir, filename)

    raise DownloadError(
        "No downloader available. Install yt-dlp (recommended) or pytube.\n"
        "pip install yt-dlp\n"
        "Optional fallback: pip install pytube"
    )


# Optional alias for backward compatibility with older imports
def download_youtube(url: str, output_dir: str = "downloads", filename: Optional[str] = None) -> str:
    return download_video(url, output_dir=output_dir, filename=filename)
