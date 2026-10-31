"""Audio I/O helpers."""
import os
import subprocess
import math

import numpy as np
import soundfile as sf


NATIVE_AUDIO_EXTS = {'.wav', '.flac', '.ogg', '.aiff', '.aif', '.au'}
FFMPEG_ONLY_EXTS = {'.m4a', '.aac', '.mp4', '.m4b', '.mp3', '.opus', '.wma'}


def needs_ffmpeg_conversion(path):
    return os.path.splitext(path)[1].lower() in FFMPEG_ONLY_EXTS


def convert_to_wav(input_path):
    out_path = input_path + ".converted.wav"
    cmd = ["ffmpeg", "-y", "-i", input_path,
           "-acodec", "pcm_s16le", out_path]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg is required to read this format, but it was not "
            "found on PATH. Install it with: sudo apt install ffmpeg "
            "(Linux) or brew install ffmpeg (macOS).")
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg conversion timed out.")

    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError("FFmpeg conversion failed: " + err[-300:])

    if not os.path.exists(out_path):
        raise RuntimeError(
            "FFmpeg reported success but no output was created.")

    return out_path


def load_audio(path):
    data, sr = sf.read(path, dtype='float32')
    if data.ndim == 1:
        data = np.column_stack((data, data))
    return data, sr


def resample_audio(data, source_rate, target_rate):
    """Convert audio to a target sample rate without changing its duration."""
    source_rate = int(source_rate)
    target_rate = int(target_rate)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive.")
    if source_rate == target_rate:
        return data
    divisor = math.gcd(source_rate, target_rate)
    from scipy.signal import resample_poly
    return resample_poly(
        data, target_rate // divisor, source_rate // divisor,
        axis=0).astype(np.float32, copy=False)


def save_wav(path, data, samplerate, subtype='PCM_24'):
    if data.ndim == 1:
        data = np.column_stack((data, data))
    sf.write(path, data, samplerate, subtype=subtype)
