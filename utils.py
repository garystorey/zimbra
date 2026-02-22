"""
Talking Head Video Generator — Utilities

File validation helpers, audio format conversion, and ffmpeg helpers.
"""

import os
import subprocess
import tempfile

import config


# ---------------------------------------------------------------------------
# Supported formats
# ---------------------------------------------------------------------------

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3")
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------


def validate_audio_file(path):
    """Validate that the file exists and is a supported audio format."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"
        )
    return path


def validate_image_file(path):
    """Validate that the file exists and is a supported image format."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}"
        )
    return path


def validate_script(text=None, script_file=None):
    """
    Validate and return script text.

    Priority: text > script_file > default.txt
    """
    if text and text.strip():
        return text.strip()

    if script_file:
        if not os.path.isfile(script_file):
            raise FileNotFoundError(f"Script file not found: {script_file}")
        with open(script_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            raise ValueError(f"Script file is empty: {script_file}")
        return content

    # Fall back to default
    default_path = config.DEFAULT_SCRIPT_FILE
    if os.path.isfile(default_path):
        with open(default_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            return content

    raise ValueError(f"No script found. Place a default.txt in {config.SCRIPTS_DIR}")


# ---------------------------------------------------------------------------
# Audio format conversion
# ---------------------------------------------------------------------------


def convert_mp3_to_wav(mp3_path, output_dir=None):
    """
    Convert an MP3 file to WAV format.

    Uses librosa/soundfile for conversion. Returns the path to the WAV file.
    If the input is already WAV, returns it unchanged.
    """
    ext = os.path.splitext(mp3_path)[1].lower()
    if ext == ".wav":
        return mp3_path

    if ext != ".mp3":
        raise ValueError(f"Expected .mp3 file, got: {ext}")

    import librosa
    import soundfile as sf

    # Load MP3
    audio, sr = librosa.load(mp3_path, sr=None, mono=False)

    # Write WAV
    if output_dir is None:
        output_dir = tempfile.gettempdir()

    basename = os.path.splitext(os.path.basename(mp3_path))[0]
    wav_path = os.path.join(output_dir, f"{basename}_converted.wav")

    sf.write(wav_path, audio.T if audio.ndim > 1 else audio, sr)
    return wav_path


def convert_wav_to_mp3(wav_path, output_path=None):
    """
    Convert a WAV file to MP3 format using FFmpeg.

    Returns the path to the MP3 file.
    """
    if output_path is None:
        basename = os.path.splitext(wav_path)[0]
        output_path = basename + ".mp3"

    ffmpeg = config.FFMPEG_PATH
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg not found. Install FFmpeg and add it to PATH, "
            "or install imageio-ffmpeg: pip install imageio-ffmpeg"
        )

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        wav_path,
        "-codec:a",
        "libmp3lame",
        "-qscale:a",
        "2",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def ensure_wav_format(audio_path, temp_dir=None):
    """
    Ensure the audio file is in WAV format for model consumption.
    Converts MP3 to WAV if needed. Returns path to a WAV file.
    """
    ext = os.path.splitext(audio_path)[1].lower()
    if ext == ".wav":
        return audio_path
    if ext == ".mp3":
        return convert_mp3_to_wav(audio_path, output_dir=temp_dir)
    raise ValueError(f"Unsupported audio format: {ext}")


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------


def ffmpeg_mux_video_audio(video_path, audio_path, output_path, fps=None):
    """
    Mux a video file with an audio file using FFmpeg.

    Args:
        video_path: Path to the video file (may have no audio or wrong audio)
        audio_path: Path to the audio file to mux in
        output_path: Path for the output file
        fps: If set, re-encode at this frame rate
    """
    ffmpeg = config.FFMPEG_PATH
    if ffmpeg is None:
        raise RuntimeError("FFmpeg not found.")

    cmd = [ffmpeg, "-y"]
    cmd += ["-i", video_path]
    cmd += ["-i", audio_path]

    if fps:
        cmd += ["-r", str(fps)]

    cmd += [
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        output_path,
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def ffmpeg_resize_video(input_path, output_path, width, height):
    """Resize a video to the specified dimensions using FFmpeg."""
    ffmpeg = config.FFMPEG_PATH
    if ffmpeg is None:
        raise RuntimeError("FFmpeg not found.")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-vf",
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-c:v",
        "libx264",
        "-c:a",
        "copy",
        output_path,
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def get_image_dimensions(image_path):
    """Get the width and height of an image using Pillow."""
    from PIL import Image

    with Image.open(image_path) as img:
        return img.size  # (width, height)


# ---------------------------------------------------------------------------
# Contact sheet generation (for quality review)
# ---------------------------------------------------------------------------


def generate_contact_sheet(video_path, output_path=None, cols=4, frame_interval=None):
    """Generate a contact sheet (grid of frames) from a video for quality review.

    Extracts frames at regular intervals and tiles them into a single PNG image.
    Useful for spotting progressive degradation, face distortion, or color drift
    across chunks without watching the full video.

    Args:
        video_path: Path to the input MP4 video.
        output_path: Where to save the PNG. Defaults to video_path with
            '_contact_sheet.png' suffix.
        cols: Number of columns in the grid (default 4).
        frame_interval: Extract every Nth frame. Defaults to ~1 frame/sec at 25fps.

    Returns:
        Path to the generated contact sheet PNG, or None on failure.
    """
    import logging

    logger = logging.getLogger("utils")

    ffmpeg = config.FFMPEG_PATH
    if ffmpeg is None:
        logger.warning("FFmpeg not found — skipping contact sheet generation")
        return None

    if not os.path.isfile(video_path):
        logger.warning(f"Video file not found for contact sheet: {video_path}")
        return None

    if output_path is None:
        base = os.path.splitext(video_path)[0]
        output_path = base + "_contact_sheet.png"

    if frame_interval is None:
        frame_interval = config.VIDEO_FPS  # ~1 frame per second

    # Probe video to get total frame count so we can size the tile grid exactly
    probe_cmd = [
        ffmpeg,
        "-i",
        video_path,
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-f",
        "null",
        "-",
    ]
    try:
        probe_result = subprocess.run(
            probe_cmd, capture_output=True, text=True, timeout=15
        )
        # Parse "frame= NNN" from stderr
        import re

        match = re.search(r"frame=\s*(\d+)", probe_result.stderr)
        total_frames = int(match.group(1)) if match else 250  # fallback ~10s
    except Exception:
        total_frames = 250

    num_extracted = max(1, total_frames // frame_interval)
    rows = max(1, (num_extracted + cols - 1) // cols)  # ceil division

    # Use FFmpeg to select every Nth frame, scale to thumbnail width, and tile
    thumb_width = 320
    vf = (
        f"select=not(mod(n\\,{frame_interval})),"
        f"scale={thumb_width}:-1,"
        f"tile={cols}x{rows}"
    )

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-vf",
        vf,
        "-vsync",
        "vfr",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning(f"FFmpeg contact sheet failed: {result.stderr[:200]}")
            return None
        if os.path.isfile(output_path):
            size_kb = os.path.getsize(output_path) / 1024
            logger.info(f"Contact sheet saved: {output_path} ({size_kb:.0f} KB)")
            return output_path
        return None
    except Exception as e:
        logger.warning(f"Contact sheet generation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# GPU thermal management
# ---------------------------------------------------------------------------


def get_gpu_temperature(gpu_index=0):
    """Read GPU temperature in °C via nvidia-smi. Returns None on failure."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
                f"--id={gpu_index}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def thermal_cooldown(
    threshold=config.GPU_THERMAL_THRESHOLD,
    target=None,
    max_wait=config.GPU_THERMAL_MAX_WAIT,
    poll_interval=config.GPU_THERMAL_POLL_INTERVAL,
    gpu_index=0,
):
    """Wait for GPU to cool below threshold before continuing.

    Args:
        threshold: Temperature (°C) above which we pause. Default from config.
        target: Temperature to cool down to (default: threshold - 5).
        max_wait: Maximum seconds to wait before giving up.
        poll_interval: Seconds between temperature checks.
        gpu_index: GPU device index.

    Returns:
        (waited_secs, start_temp, end_temp) or (0, temp, temp) if no wait needed.
    """
    import logging
    import time

    logger = logging.getLogger("utils")

    if target is None:
        target = threshold - 5

    temp = get_gpu_temperature(gpu_index)
    if temp is None:
        return (0, None, None)

    if temp <= threshold:
        return (0, temp, temp)

    logger.info(
        f"[thermal] GPU at {temp}°C (threshold {threshold}°C), "
        f"cooling to {target}°C (max {max_wait}s wait)..."
    )

    start_temp = temp
    waited = 0.0
    while temp > target and waited < max_wait:
        time.sleep(poll_interval)
        waited += poll_interval
        temp = get_gpu_temperature(gpu_index)
        if temp is None:
            break

    end_temp = temp if temp is not None else start_temp
    if waited > 0:
        logger.info(f"[thermal] Cooled {start_temp}°C → {end_temp}°C in {waited:.0f}s")
    return (waited, start_temp, end_temp)
