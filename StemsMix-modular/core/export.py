"""Export: render mix to temp WAV, encode with FFmpeg."""
import shutil
import subprocess

import numpy as np
import soundfile as sf

from .audio_io import save_wav


EXPORT_FORMATS = {
    "WAV (lossless)": {
        "ext": "wav",
        "filter": "WAV Files (*.wav)",
        "bitrates": [("Lossless (24-bit PCM)", None)],
    },
    "MP3": {
        "ext": "mp3",
        "filter": "MP3 Files (*.mp3)",
        "bitrates": [("128 kbps", "128k"), ("192 kbps", "192k"),
                     ("256 kbps", "256k"), ("320 kbps", "320k")],
    },
    "M4A (AAC)": {
        "ext": "m4a",
        "filter": "M4A Files (*.m4a)",
        "bitrates": [("128 kbps", "128k"), ("192 kbps", "192k"),
                     ("256 kbps", "256k")],
    },
    "AAC": {
        "ext": "aac",
        "filter": "AAC Files (*.aac)",
        "bitrates": [("128 kbps", "128k"), ("192 kbps", "192k"),
                     ("256 kbps", "256k")],
    },
}

QUALITY_NOTES = {
    ("WAV (lossless)", "Lossless (24-bit PCM)"):
        "Uncompressed 24-bit PCM. Large files.",
    ("MP3", "128 kbps"): "Small files, audible loss.",
    ("MP3", "192 kbps"): "Good balance of size and quality.",
    ("MP3", "256 kbps"): "Near-transparent.",
    ("MP3", "320 kbps"): "Maximum MP3 quality.",
    ("M4A (AAC)", "128 kbps"): "Small and efficient.",
    ("M4A (AAC)", "192 kbps"): "Very good quality.",
    ("M4A (AAC)", "256 kbps"): "Excellent quality.",
    ("AAC", "128 kbps"): "Raw AAC stream, small files.",
    ("AAC", "192 kbps"): "Raw AAC stream, very good quality.",
    ("AAC", "256 kbps"): "Raw AAC stream, excellent quality.",
}


def render_mix_to_wav(tracks, dsp_chains, wav_path,
                      progress_cb=None):
    active = [(t, d) for t, d in zip(tracks, dsp_chains)
              if not t.muted]
    if not active:
        return False

    max_len = max(len(t.data) for t, _ in active)
    sr = active[0][0].samplerate
    mix = np.zeros((max_len, 2), dtype='float32')

    for i, (track, dsp) in enumerate(active):
        processed = dsp.process_offline(track.data * track.volume)
        mix[:len(processed)] += processed
        if progress_cb:
            progress_cb(int((i + 1) / len(active) * 100))

    mix = np.clip(mix, -1.0, 1.0)
    save_wav(wav_path, mix, sr)
    if progress_cb:
        progress_cb(100)
    return True


def encode_with_ffmpeg(input_wav, output_path, fmt_key, bitrate,
                       progress_cb=None):
    if fmt_key == "WAV (lossless)":
        if input_wav != output_path:
            shutil.copyfile(input_wav, output_path)
        if progress_cb:
            progress_cb(100)
        return True

    if fmt_key == "MP3":
        codec = ["-codec:a", "libmp3lame", "-b:a", bitrate]
    elif fmt_key == "M4A (AAC)":
        codec = ["-codec:a", "aac", "-b:a", bitrate]
    elif fmt_key == "AAC":
        codec = ["-codec:a", "aac", "-b:a", bitrate, "-f", "adts"]
    else:
        return False

    cmd = ["ffmpeg", "-y", "-i", input_wav,
           "-progress", "pipe:1", "-nostats"] + codec + [output_path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    try:
        info = sf.info(input_wav)
        total_us = info.frames / info.samplerate * 1_000_000
    except Exception:
        total_us = 0

    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_ms=") and total_us > 0:
            try:
                us = int(line.split("=", 1)[1])
                if progress_cb:
                    progress_cb(
                        max(0, min(100, int(us / total_us * 100))))
            except ValueError:
                pass

    proc.wait()
    if proc.returncode != 0:
        return False
    if progress_cb:
        progress_cb(100)
    return True
