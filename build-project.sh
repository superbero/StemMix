#!/usr/bin/env bash
# build-project.sh
# Creates the StemsMix modular project and packages it as a zip.

set -euo pipefail

ROOT="StemsMix-modular"
ZIP_NAME="StemsMix-modular.zip"

echo ">> Cleaning previous build..."
rm -rf "$ROOT"
rm -f "$ZIP_NAME"

echo ">> Creating directories..."
mkdir -p "$ROOT/core" "$ROOT/ui" "$ROOT/apps"

# ===========================================================================
# ROOT FILES
# ===========================================================================

echo ">> requirements.txt"
cat > "$ROOT/requirements.txt" <<'FILE_END'
demucs-onnx>=0.3.4
numpy>=1.24,<2.0
scipy>=1.11
psutil>=5.9
PyQt6>=6.6
sounddevice>=0.4.6
soundfile>=0.12
FILE_END

echo ">> .gitignore"
cat > "$ROOT/.gitignore" <<'FILE_END'
.venv/
venv/
env/
__pycache__/
*.py[cod]
*.so
build/
dist/
*.egg-info/
.cache/
*.onnx
*.pt
*.pth
*.gguf
*.wav
*.mp3
*.m4a
*.mp4
*.m4b
*.aac
*.flac
*.ogg
*_stems/
.vscode/
.idea/
*.swp
.DS_Store
Thumbs.db
*.tmp
FILE_END

echo ">> README.md"
cat > "$ROOT/README.md" <<'FILE_END'
# StemsMix

Stem separation, mixing, and DAW-style timeline for Linux and macOS.

## Structure

- `core/` — pure Python, no Qt. DSP, audio I/O, separation, export.
- `ui/`   — Qt widgets and dialogs.
- `apps/` — thin entry points: `mixer_app.py` and `daw.py`.

## Setup

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Linux: sudo apt install portaudio19-dev ffmpeg
macOS: brew install portaudio ffmpeg

## Run

    python apps/mixer_app.py
    python apps/daw.py
FILE_END

# ===========================================================================
# CORE PACKAGE
# ===========================================================================

echo ">> core/__init__.py"
cat > "$ROOT/core/__init__.py" <<'FILE_END'
"""Core logic for StemsMix. No Qt imports allowed in this package."""
FILE_END

echo ">> core/resources.py"
cat > "$ROOT/core/resources.py" <<'FILE_END'
"""System prerequisite checks and resource probing."""
import os
import subprocess
from dataclasses import dataclass

import psutil

try:
    import sounddevice as sd
    SOUNDDEVICE_IMPORT_ERROR = None
except OSError as e:
    sd = None
    SOUNDDEVICE_IMPORT_ERROR = str(e)

try:
    import demucs_onnx
    DEMUCS_IMPORT_ERROR = None
except ImportError as e:
    demucs_onnx = None
    DEMUCS_IMPORT_ERROR = str(e)


MIN_CPU_CORES_FAIL = 2
MIN_CPU_CORES_WARN = 4
MIN_RAM_GB_FAIL = 4
MIN_RAM_GB_WARN = 8
MIN_DISK_GB_FAIL = 1
MIN_DISK_GB_WARN = 5
MIN_RAM_6_STEM_GB = 6
MIN_CORES_6_STEM = 4


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    blocking: bool


def ffmpeg_available():
    try:
        r = subprocess.run(["ffmpeg", "-version"],
                           capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_prerequisite_checks():
    results = []

    if SOUNDDEVICE_IMPORT_ERROR:
        results.append(CheckResult(
            "PortAudio", "fail",
            "Audio playback unavailable: {}. "
            "Install with: sudo apt install portaudio19-dev".format(
                SOUNDDEVICE_IMPORT_ERROR),
            True))
    else:
        results.append(CheckResult("PortAudio", "pass", "Available", False))

    if DEMUCS_IMPORT_ERROR:
        results.append(CheckResult(
            "demucs-onnx", "fail",
            "Separation engine unavailable: {}. "
            "Install with: pip install demucs-onnx".format(
                DEMUCS_IMPORT_ERROR),
            True))
    else:
        results.append(CheckResult("demucs-onnx", "pass",
                                   "Available", False))

    if ffmpeg_available():
        results.append(CheckResult("FFmpeg", "pass",
                                   "Found on PATH", False))
    else:
        results.append(CheckResult(
            "FFmpeg", "warn",
            "Not found - MP3/AAC/M4A input and export will not work.",
            False))

    try:
        cores = psutil.cpu_count(logical=True) or 1
    except Exception:
        cores = 1

    if cores >= MIN_CPU_CORES_WARN:
        results.append(CheckResult("CPU", "pass",
                                   "{} logical cores".format(cores), False))
    elif cores >= MIN_CPU_CORES_FAIL:
        results.append(CheckResult(
            "CPU", "warn",
            "Only {} cores - separation will be slow".format(cores),
            False))
    else:
        results.append(CheckResult(
            "CPU", "fail",
            "Only {} core - at least {} required".format(
                cores, MIN_CPU_CORES_FAIL),
            True))

    try:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        avail_gb = mem.available / (1024 ** 3)
    except Exception:
        total_gb, avail_gb = 0.0, 0.0

    ram_msg = "{:.1f} GB total, {:.1f} GB available".format(
        total_gb, avail_gb)
    if total_gb >= MIN_RAM_GB_WARN:
        results.append(CheckResult("RAM", "pass", ram_msg, False))
    elif total_gb >= MIN_RAM_GB_FAIL:
        results.append(CheckResult(
            "RAM", "warn",
            ram_msg + " - 6-stem separation may freeze the system",
            False))
    else:
        results.append(CheckResult(
            "RAM", "fail",
            ram_msg + " - at least {} GB required".format(MIN_RAM_GB_FAIL),
            True))

    try:
        disk = psutil.disk_usage(os.path.expanduser("~"))
        free_gb = disk.free / (1024 ** 3)
    except Exception:
        free_gb = 0.0

    disk_msg = "{:.1f} GB free in home directory".format(free_gb)
    if free_gb >= MIN_DISK_GB_WARN:
        results.append(CheckResult("Disk space", "pass", disk_msg, False))
    elif free_gb >= MIN_DISK_GB_FAIL:
        results.append(CheckResult(
            "Disk space", "warn",
            disk_msg + " - model download may fail", False))
    else:
        results.append(CheckResult(
            "Disk space", "fail",
            disk_msg + " - at least {} GB required".format(
                MIN_DISK_GB_FAIL),
            True))

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            results.append(CheckResult(
                "GPU", "info", "CUDA available", False))
        elif "CoreMLExecutionProvider" in providers:
            results.append(CheckResult(
                "GPU", "info", "CoreML available (Apple Silicon)", False))
        else:
            results.append(CheckResult(
                "GPU", "info", "CPU only (no GPU acceleration)", False))
    except ImportError:
        results.append(CheckResult(
            "GPU", "info", "onnxruntime not installed", False))

    return results


def check_system_resources():
    try:
        mem = psutil.virtual_memory()
        ram_gb = mem.available / (1024 ** 3)
        cores = psutil.cpu_count(logical=True) or 1
    except Exception:
        return {
            "ram_gb": 0.0, "cores": 0, "can_run_6_stem": False,
            "reason": "Could not determine system resources.",
        }

    can_run = True
    reason = None
    if ram_gb < MIN_RAM_6_STEM_GB:
        can_run = False
        reason = "Only {:.1f} GB RAM available; 6-stem needs {} GB.".format(
            ram_gb, MIN_RAM_6_STEM_GB)
    elif cores < MIN_CORES_6_STEM:
        can_run = False
        reason = "Only {} CPU core(s); 6-stem needs at least {}.".format(
            cores, MIN_CORES_6_STEM)

    return {
        "ram_gb": ram_gb, "cores": cores,
        "can_run_6_stem": can_run, "reason": reason,
    }
FILE_END

echo ">> core/audio_io.py"
cat > "$ROOT/core/audio_io.py" <<'FILE_END'
"""Audio I/O helpers."""
import os
import subprocess

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


def save_wav(path, data, samplerate, subtype='PCM_24'):
    if data.ndim == 1:
        data = np.column_stack((data, data))
    sf.write(path, data, samplerate, subtype=subtype)
FILE_END

echo ">> core/dsp.py"
cat > "$ROOT/core/dsp.py" <<'FILE_END'
"""Pure-numpy DSP: EQ, compressor, reverb."""
import math

import numpy as np
from scipy import signal as sp_signal


class DSPChain:
    """3-band EQ, compressor, reverb. Neutral by default."""

    def __init__(self, samplerate):
        self.sr = samplerate
        self.eq_bass = 0.0
        self.eq_mid = 0.0
        self.eq_treble = 0.0
        self.comp_threshold = 0.0
        self.comp_ratio = 4.0
        self.comp_attack = 0.01
        self.comp_release = 0.1
        self.reverb_mix = 0.0
        self._filters_dirty = True
        self._bass_sos = None
        self._mid_sos = None
        self._treble_sos = None
        self._bass_zi = None
        self._mid_zi = None
        self._treble_zi = None
        self._comp_env = 0.0
        self._reverb_buffers = None

    def _rebuild_filters(self):
        sr = self.sr

        A = 10 ** (self.eq_bass / 40.0)
        w0 = 2 * math.pi * 250 / sr
        alpha = math.sin(w0) / 2 * math.sqrt(
            (A + 1 / A) * (1 / 0.707 - 1) + 2)
        b0 = A * ((A + 1) - (A - 1) * math.cos(w0)
                  + 2 * math.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * math.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * math.cos(w0)
                  - 2 * math.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * math.cos(w0))
        a2 = (A + 1) + (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha
        self._bass_sos = np.array(
            [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])

        A = 10 ** (self.eq_mid / 40.0)
        w0 = 2 * math.pi * 1000 / sr
        alpha = math.sin(w0) / 2
        b0 = 1 + alpha * A
        b1 = -2 * math.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * math.cos(w0)
        a2 = 1 - alpha / A
        self._mid_sos = np.array(
            [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])

        A = 10 ** (self.eq_treble / 40.0)
        w0 = 2 * math.pi * 4000 / sr
        alpha = math.sin(w0) / 2 * math.sqrt(
            (A + 1 / A) * (1 / 0.707 - 1) + 2)
        b0 = A * ((A + 1) + (A - 1) * math.cos(w0)
                  + 2 * math.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * math.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * math.cos(w0)
                  - 2 * math.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * math.cos(w0))
        a2 = (A + 1) - (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha
        self._treble_sos = np.array(
            [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])

        self._bass_zi = np.zeros((1, 2))
        self._mid_zi = np.zeros((1, 2))
        self._treble_zi = np.zeros((1, 2))
        self._filters_dirty = False

    def process(self, audio):
        if audio.shape[0] == 0:
            return audio
        if (self.eq_bass == 0 and self.eq_mid == 0
                and self.eq_treble == 0 and self.comp_threshold == 0
                and self.reverb_mix == 0):
            return audio

        if self._filters_dirty:
            self._rebuild_filters()

        out = audio.copy()
        if self.eq_bass or self.eq_mid or self.eq_treble:
            for ch in range(out.shape[1]):
                if self.eq_bass:
                    out[:, ch], self._bass_zi[0] = sp_signal.sosfilt(
                        self._bass_sos, out[:, ch],
                        zi=self._bass_zi[0])
                if self.eq_mid:
                    out[:, ch], self._mid_zi[0] = sp_signal.sosfilt(
                        self._mid_sos, out[:, ch],
                        zi=self._mid_zi[0])
                if self.eq_treble:
                    out[:, ch], self._treble_zi[0] = sp_signal.sosfilt(
                        self._treble_sos, out[:, ch],
                        zi=self._treble_zi[0])

        if self.comp_threshold < 0 and self.comp_ratio > 1.0:
            thresh_lin = 10 ** (self.comp_threshold / 20.0)
            attack_coeff = math.exp(
                -1.0 / (self.comp_attack * self.sr))
            release_coeff = math.exp(
                -1.0 / (self.comp_release * self.sr))
            for i in range(out.shape[0]):
                peak = float(np.max(np.abs(out[i]))) + 1e-12
                if peak > self._comp_env:
                    self._comp_env = (attack_coeff * self._comp_env
                                      + (1 - attack_coeff) * peak)
                else:
                    self._comp_env = (release_coeff * self._comp_env
                                      + (1 - release_coeff) * peak)
                if self._comp_env > thresh_lin:
                    over_db = 20 * math.log10(
                        self._comp_env / thresh_lin)
                    gain_db = -over_db * (1 - 1.0 / self.comp_ratio)
                    out[i] *= 10 ** (gain_db / 20.0)

        if self.reverb_mix > 0:
            if self._reverb_buffers is None:
                self._reverb_buffers = self._make_reverb_buffers()
            wet = np.zeros_like(out)
            for ch in range(out.shape[1]):
                wet[:, ch] = self._reverb_channel(out[:, ch], ch)
            out = (1 - self.reverb_mix) * out + self.reverb_mix * wet

        return out

    def _make_reverb_buffers(self):
        comb_delays = [1116, 1188, 1277, 1356]
        allpass_delays = [556, 441]
        scale = self.sr / 44100.0
        buffers = []
        for ch in range(2):
            ch_bufs = []
            for d in comb_delays:
                n = max(1, int(d * scale))
                ch_bufs.append({"buf": np.zeros(n),
                                "pos": 0, "filter": 0.0})
            for d in allpass_delays:
                n = max(1, int(d * scale))
                ch_bufs.append({"buf": np.zeros(n), "pos": 0})
            buffers.append(ch_bufs)
        return buffers

    def _reverb_channel(self, x, ch):
        bufs = self._reverb_buffers[ch]
        feedback = 0.84
        damp = 0.4
        wet = np.zeros_like(x)
        for i in range(len(x)):
            sample = x[i]
            acc = 0.0
            for c in bufs[:4]:
                buf = c["buf"]
                pos = c["pos"]
                out = buf[pos]
                c["filter"] = out * (1 - damp) + c["filter"] * damp
                buf[pos] = sample + c["filter"] * feedback
                c["pos"] = (pos + 1) % len(buf)
                acc += out
            acc *= 0.25
            for a in bufs[4:]:
                buf = a["buf"]
                pos = a["pos"]
                buf_out = buf[pos]
                buf[pos] = acc + buf_out * 0.5
                a["pos"] = (pos + 1) % len(buf)
                acc = buf_out - acc
            wet[i] = acc
        return wet

    def process_offline(self, audio):
        self._filters_dirty = True
        self._comp_env = 0.0
        self._reverb_buffers = None
        return self.process(audio)
FILE_END

echo ">> core/separation.py"
cat > "$ROOT/core/separation.py" <<'FILE_END'
"""Separation engine (demucs-onnx)."""
import os
import shutil

import numpy as np
import soundfile as sf

from .audio_io import convert_to_wav, needs_ffmpeg_conversion
from .resources import DEMUCS_IMPORT_ERROR

if DEMUCS_IMPORT_ERROR is None:
    import demucs_onnx
else:
    demucs_onnx = None


def separate_file(input_path, output_dir, model_name,
                  providers="auto", precision="fp32",
                  chunked=False, chunk_seconds=60, progress_cb=None):
    if DEMUCS_IMPORT_ERROR:
        raise RuntimeError("demucs-onnx unavailable: "
                           + DEMUCS_IMPORT_ERROR)

    def emit(pct, msg):
        if progress_cb:
            progress_cb(int(pct), str(msg))

    converted = None
    input_to_use = input_path
    try:
        if needs_ffmpeg_conversion(input_path):
            emit(1, "Converting input to WAV...")
            try:
                converted = convert_to_wav(input_path)
                input_to_use = converted
                emit(5, "Conversion complete.")
            except RuntimeError as e:
                raise RuntimeError(str(e))

        os.makedirs(output_dir, exist_ok=True)

        if chunked:
            return _run_chunked(input_to_use, output_dir, model_name,
                                providers, precision, chunk_seconds, emit)
        return _run_single_pass(input_to_use, output_dir, model_name,
                                providers, precision, emit)
    finally:
        if converted and os.path.exists(converted):
            try:
                os.remove(converted)
            except OSError:
                pass


def _run_single_pass(input_path, output_dir, model_name,
                     providers, precision, emit):
    emit(10, "Loading model and separating...")
    kwargs = dict(model=model_name, providers=providers,
                  precision=precision, progress=False)
    emit(20, "Running separation (this may take a while)...")
    result = demucs_onnx.separate(input_path, output_dir, **kwargs)
    emit(95, "Finalizing...")
    normalized = _normalize_result(result)
    emit(100, "Done")
    return normalized


def _run_chunked(input_path, output_dir, model_name, providers,
                 precision, chunk_seconds, emit):
    emit(10, "Reading audio file...")
    audio, sr = sf.read(input_path, dtype="float32", always_2d=True)
    total_frames = audio.shape[0]
    chunk_frames = int(chunk_seconds * sr)
    n_chunks = max(1, (total_frames + chunk_frames - 1) // chunk_frames)

    stem_chunks = {}
    tmp_dir = os.path.join(output_dir, "_tmp_chunks")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        for i in range(n_chunks):
            start = i * chunk_frames
            end = min(start + chunk_frames, total_frames)
            chunk = audio[start:end]
            chunk_path = os.path.join(tmp_dir,
                                      "chunk_{:04d}.wav".format(i))
            sf.write(chunk_path, chunk, sr)

            pct = 10 + int((i / n_chunks) * 80)
            emit(pct, "Separating chunk {}/{}...".format(
                i + 1, n_chunks))

            kwargs = dict(model=model_name, providers=providers,
                          precision=precision, progress=False)
            result = demucs_onnx.separate(chunk_path, tmp_dir, **kwargs)

            for stem_name, stem_path in _normalize_result(
                    result).items():
                data, _ = sf.read(stem_path, dtype="float32",
                                  always_2d=True)
                stem_chunks.setdefault(stem_name, []).append(data)

            try:
                os.remove(chunk_path)
            except OSError:
                pass

        emit(92, "Writing final stems...")
        out_paths = {}
        for stem_name, chunks in stem_chunks.items():
            combined = np.concatenate(chunks, axis=0)
            out_path = os.path.join(output_dir, stem_name + ".wav")
            sf.write(out_path, combined, sr)
            out_paths[stem_name] = out_path

        emit(100, "Done")
        return out_paths
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def _normalize_result(result):
    if isinstance(result, dict):
        return {k: v for k, v in result.items()
                if isinstance(v, str) and os.path.exists(v)}
    if isinstance(result, (list, tuple)):
        out = {}
        for item in result:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                name, path = item
                if isinstance(path, str) and os.path.exists(path):
                    out[str(name)] = path
            elif isinstance(item, str) and os.path.exists(item):
                name = os.path.splitext(os.path.basename(item))[0]
                out[name] = item
        return out
    return {}
FILE_END

echo ">> core/export.py"
cat > "$ROOT/core/export.py" <<'FILE_END'
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
FILE_END

# ===========================================================================
# UI PACKAGE
# ===========================================================================

echo ">> ui/__init__.py"
cat > "$ROOT/ui/__init__.py" <<'FILE_END'
"""Qt UI layer for StemsMix."""
FILE_END

echo ">> ui/style.py"
cat > "$ROOT/ui/style.py" <<'FILE_END'
"""QSS stylesheet and color palette."""
COLORS = {
    "bg": "#1e1e22",
    "bg_alt": "#26262c",
    "bg_card": "#2c2c34",
    "bg_hover": "#35353e",
    "border": "#3a3a44",
    "text": "#e6e6ea",
    "text_dim": "#9a9aa4",
    "accent": "#4a9eff",
    "accent_hover": "#6cb0ff",
    "success": "#5ed17a",
    "warning": "#f0b655",
    "danger": "#f06060",
    "track_colors": {
        "vocals": "#4a9eff",
        "drums": "#f09055",
        "bass": "#5ed17a",
        "other": "#b07aff",
        "guitar": "#f06c9c",
        "piano": "#f0d055",
    },
}

QSS = """
QWidget {
    background-color: %(bg)s;
    color: %(text)s;
    font-family: -apple-system, "Segoe UI", "Ubuntu", "Helvetica Neue", sans-serif;
    font-size: 14px;
}
QMainWindow { background-color: %(bg)s; }
QLabel { background: transparent; color: %(text)s; }
QLabel#header_title { font-size: 20px; font-weight: 600; }
QLabel#header_subtitle { font-size: 12px; color: %(text_dim)s; }
QLabel#status_bar {
    background-color: %(bg_alt)s; color: %(text_dim)s;
    padding: 8px 14px; border-radius: 6px; font-size: 12px;
}
QLabel#progress_label { color: %(text_dim)s; font-size: 12px; }
QLabel#position_label {
    color: %(text_dim)s; font-size: 12px;
    font-family: "Menlo", "Consolas", monospace;
}
QPushButton {
    background-color: %(bg_card)s; color: %(text)s;
    border: 1px solid %(border)s; border-radius: 6px;
    padding: 8px 16px; font-size: 13px; font-weight: 500;
}
QPushButton:hover {
    background-color: %(bg_hover)s; border-color: %(accent)s;
}
QPushButton:pressed { background-color: %(bg_alt)s; }
QPushButton:disabled {
    color: %(text_dim)s; background-color: %(bg_alt)s;
}
QPushButton#primary {
    background-color: %(accent)s; color: #ffffff;
    border: none; font-weight: 600; padding: 9px 20px;
}
QPushButton#primary:hover { background-color: %(accent_hover)s; }
QPushButton#primary:disabled {
    background-color: %(border)s; color: %(text_dim)s;
}
QSlider::groove:horizontal {
    height: 4px; background: %(border)s; border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: %(accent)s; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #ffffff; width: 14px; height: 14px;
    margin: -6px 0; border-radius: 7px;
    border: 2px solid %(accent)s;
}
QProgressBar {
    background-color: %(bg_alt)s; border: none;
    border-radius: 6px; height: 10px;
    text-align: center; color: transparent;
}
QProgressBar::chunk {
    background-color: %(accent)s; border-radius: 6px;
}
QGroupBox {
    background-color: %(bg_alt)s; border: 1px solid %(border)s;
    border-radius: 10px; margin-top: 14px;
    padding-top: 22px; font-weight: 600;
    font-size: 13px; color: %(text_dim)s;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 16px; padding: 0 6px; font-size: 11px;
    color: %(text_dim)s;
}
QCheckBox {
    background: transparent; spacing: 8px;
    font-size: 12px; color: %(text_dim)s;
}
QCheckBox::indicator {
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid %(border)s; background: %(bg)s;
}
QCheckBox::indicator:checked {
    background: %(accent)s; border-color: %(accent)s;
}
QComboBox {
    background-color: %(bg_card)s; border: 1px solid %(border)s;
    border-radius: 6px; padding: 7px 12px;
    font-size: 13px; min-width: 180px;
}
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background-color: %(bg_card)s;
    border: 1px solid %(border)s; border-radius: 6px;
    selection-background-color: %(accent)s;
    padding: 4px; outline: none;
}
QDialog { background-color: %(bg)s; }
QScrollArea { background: transparent; border: none; }
""" % COLORS
FILE_END

echo ">> ui/dialogs.py"
cat > "$ROOT/ui/dialogs.py" <<'FILE_END'
"""Dialogs: prerequisites, model selection, export."""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                             QLabel, QDialogButtonBox, QComboBox,
                             QCheckBox, QFormLayout, QScrollArea,
                             QWidget)

from .style import COLORS
from core.export import EXPORT_FORMATS, QUALITY_NOTES


class PrerequisiteDialog(QDialog):
    STATUS_ICON = {"pass": "OK", "warn": "!", "fail": "X", "info": "i"}

    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results
        self.setWindowTitle("System Prerequisites")
        self.setMinimumWidth(640)
        self.has_blocking_failure = any(
            r.blocking and r.status == "fail" for r in results)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        if self.has_blocking_failure:
            header_text = ("Cannot start. Fix the items marked FAIL.")
            bg_style = ("background-color: #3a1e22; "
                        "color: #f8b6be; border: 1px solid #5a2a30;")
        else:
            header_text = ("Some warnings detected. The app should work, "
                           "but some features may be limited.")
            bg_style = ("background-color: #3a3020; "
                        "color: #f0d090; border: 1px solid #5a4a20;")

        header = QLabel(header_text)
        header.setWordWrap(True)
        header.setStyleSheet(
            "QLabel { padding: 14px; border-radius: 8px; "
            "font-size: 13px; " + bg_style + " }")
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(360)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(6)
        for r in results:
            inner_layout.addWidget(self._make_row(r))
        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox()
        if self.has_blocking_failure:
            buttons.addButton(
                "Quit", QDialogButtonBox.ButtonRole.RejectRole)
        else:
            cont = buttons.addButton(
                "Continue anyway",
                QDialogButtonBox.ButtonRole.AcceptRole)
            cont.setObjectName("primary")
            cont.setDefault(True)
            buttons.addButton(
                "Quit", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _make_row(self, r):
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(10)
        icon = QLabel(self.STATUS_ICON.get(r.status, "?"))
        icon.setFixedWidth(28)
        icon.setStyleSheet("font-weight: 700; font-size: 14px;")
        row.addWidget(icon)
        name = QLabel(r.name)
        name.setFixedWidth(110)
        row.addWidget(name)
        msg = QLabel(r.message)
        msg.setWordWrap(True)
        msg.setStyleSheet(
            "color: " + COLORS["text_dim"] + "; font-size: 12px;")
        row.addWidget(msg, 1)
        bg = {"pass": "#1e3a24", "warn": "#3a3020",
              "fail": "#3a1e22", "info": "#2a2a30"}.get(
                  r.status, COLORS["bg_card"])
        row_widget.setStyleSheet(
            "QWidget { background-color: " + bg +
            "; border-radius: 6px; }"
            "QLabel { background-color: transparent; }")
        return row_widget


class ModelSelectionDialog(QDialog):
    def __init__(self, resources, parent=None):
        super().__init__(parent)
        self.resources = resources
        self.setWindowTitle("Choose Separation Model")
        self.setMinimumWidth(620)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("Separation Model")
        title.setObjectName("header_title")
        layout.addWidget(title)

        info = QLabel(
            "High Quality gives the best results. Fast is roughly 30% "
            "faster with slightly lower quality. 6-stem adds guitar "
            "and piano at the cost of higher memory use.")
        info.setWordWrap(True)
        info.setStyleSheet(
            "color: " + COLORS["text_dim"] + "; font-size: 13px;")
        layout.addWidget(info)

        sys_line = QLabel(
            "Detected: {:.1f} GB available RAM, {} CPU core(s).".format(
                resources["ram_gb"], resources["cores"]))
        sys_line.setStyleSheet(
            "color: " + COLORS["text_dim"] + "; font-size: 12px;")
        layout.addWidget(sys_line)

        self.combo = QComboBox()
        self.combo.addItem(
            "High Quality 4-stem (htdemucs_ft)", "htdemucs_ft")
        self.combo.addItem(
            "Fast 4-stem (htdemucs)", "htdemucs")
        if resources["can_run_6_stem"]:
            self.combo.addItem(
                "Fine 6-stem (htdemucs_6s)", "htdemucs_6s")
        self.combo.currentIndexChanged.connect(self.on_model_changed)
        layout.addWidget(self.combo)

        self.force_cpu_cb = QCheckBox("Force CPU")
        layout.addWidget(self.force_cpu_cb)

        self.chunked_cb = QCheckBox(
            "Use chunked separation (lower peak memory)")
        self.chunked_cb.setChecked(resources["ram_gb"] < 8)
        layout.addWidget(self.chunked_cb)

        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet(
            "QLabel { background-color: #3a3020; color: #f0d090; "
            "padding: 12px; border: 1px solid #5a4a20; "
            "border-radius: 6px; font-size: 12px; }")
        layout.addWidget(self.warning_label)

        if not resources["can_run_6_stem"]:
            blocked = QLabel(
                "6-stem mode is disabled: " +
                (resources["reason"] or ""))
            blocked.setWordWrap(True)
            blocked.setStyleSheet(
                "QLabel { background-color: #3a1e22; color: #f8b6be; "
                "padding: 12px; border: 1px solid #5a2a30; "
                "border-radius: 6px; font-size: 12px; }")
            layout.addWidget(blocked)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        ok_btn = buttons.button(
            QDialogButtonBox.StandardButton.Ok)
        if ok_btn:
            ok_btn.setObjectName("primary")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.on_model_changed()

    def on_model_changed(self):
        model_name = self.combo.currentData()
        if model_name == "htdemucs_6s":
            self.force_cpu_cb.setChecked(True)
            self.warning_label.setText(
                "6-stem is RAM-heavy. Force CPU is pre-selected.")
        elif model_name == "htdemucs":
            self.force_cpu_cb.setChecked(False)
            self.warning_label.setText(
                "Fast 4-stem: single file, ~30% faster.")
        else:
            self.force_cpu_cb.setChecked(False)
            self.warning_label.setText(
                "High Quality 4-stem: best separation quality.")

    def selected_model(self):
        return self.combo.currentData()

    def force_cpu(self):
        return self.force_cpu_cb.isChecked()

    def use_chunked(self):
        return self.chunked_cb.isChecked()


class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Mix")
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("Export Mix")
        title.setObjectName("header_title")
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(10)
        self.format_combo = QComboBox()
        self.format_combo.addItems(EXPORT_FORMATS.keys())
        self.format_combo.currentTextChanged.connect(
            self.on_format_changed)
        form.addRow("Format:", self.format_combo)
        self.bitrate_combo = QComboBox()
        form.addRow("Bitrate:", self.bitrate_combo)
        layout.addLayout(form)

        self.quality_label = QLabel()
        self.quality_label.setWordWrap(True)
        self.quality_label.setMinimumHeight(60)
        self.quality_label.setStyleSheet(
            "QLabel { background-color: " + COLORS["bg_card"] + "; "
            "color: " + COLORS["text_dim"] + "; padding: 12px; "
            "border: 1px solid " + COLORS["border"] + "; "
            "border-radius: 6px; font-size: 12px; }")
        layout.addWidget(self.quality_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        ok_btn = buttons.button(
            QDialogButtonBox.StandardButton.Ok)
        if ok_btn:
            ok_btn.setObjectName("primary")
            ok_btn.setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.on_format_changed(self.format_combo.currentText())

    def on_format_changed(self, fmt_key):
        self.bitrate_combo.blockSignals(True)
        self.bitrate_combo.clear()
        for label, value in EXPORT_FORMATS[fmt_key]["bitrates"]:
            self.bitrate_combo.addItem(label, value)
        self.bitrate_combo.blockSignals(False)
        self.bitrate_combo.currentIndexChanged.connect(
            self.update_quality_label)
        self.update_quality_label()

    def update_quality_label(self):
        fmt = self.format_combo.currentText()
        br = self.bitrate_combo.currentText()
        self.quality_label.setText(
            QUALITY_NOTES.get((fmt, br), ""))

    def selected_format(self):
        return self.format_combo.currentText()

    def selected_bitrate(self):
        return self.bitrate_combo.currentData()
FILE_END

echo ">> ui/timeline.py"
cat > "$ROOT/ui/timeline.py" <<'FILE_END'
"""Timeline scene/view/clip widgets."""
from PyQt6.QtWidgets import (QGraphicsView, QGraphicsScene,
                             QGraphicsItem, QGraphicsRectItem,
                             QGraphicsLineItem, QGraphicsTextItem)
from PyQt6.QtGui import QBrush, QPen, QColor, QFont, QPainter
from PyQt6.QtCore import Qt, QPointF, QWheelEvent

from .style import COLORS


TRACK_HEIGHT = 90
PIXELS_PER_SECOND = 80


class ClipItem(QGraphicsRectItem):
    def __init__(self, stem_name, duration, track_index,
                 parent=None):
        h = TRACK_HEIGHT - 16
        w = max(20, duration * PIXELS_PER_SECOND)
        super().__init__(0, 0, w, h, parent)
        self.stem_name = stem_name
        self.track_index = track_index
        self.duration = duration

        color = QColor(COLORS["track_colors"].get(
            stem_name, COLORS["accent"]))
        self.setBrush(QBrush(color.darker(160)))
        self.setPen(QPen(color, 2))
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
            True)
        self.setY(track_index * TRACK_HEIGHT + 8)
        self.setX(8)

        label = QGraphicsTextItem(stem_name.capitalize(), self)
        label.setDefaultTextColor(QColor("#ffffff"))
        f = QFont()
        f.setBold(True)
        f.setPointSize(10)
        label.setFont(f)
        label.setPos(8, 4)

        dur_text = "{}:{:02d}".format(
            int(duration // 60), int(duration % 60))
        dl = QGraphicsTextItem(dur_text, self)
        dl.setDefaultTextColor(QColor("#cccccc"))
        f2 = QFont()
        f2.setPointSize(8)
        dl.setFont(f2)
        dl.setPos(8, h - 18)

    def itemChange(self, change, value):
        if change == \
                QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            new_pos = value
            lane = max(0, round(new_pos.y() / TRACK_HEIGHT))
            return QPointF(max(0, new_pos.x()),
                           lane * TRACK_HEIGHT + 8)
        if change == \
                QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.track_index = round(self.y() / TRACK_HEIGHT)
        return super().itemChange(change, value)


class TimelineScene(QGraphicsScene):
    def __init__(self, num_tracks=6, parent=None):
        super().__init__(parent)
        self.num_tracks = num_tracks
        self.setBackgroundBrush(QBrush(QColor(COLORS["bg"])))
        self.setSceneRect(
            0, 0, 10000, num_tracks * TRACK_HEIGHT + 20)

        for i in range(num_tracks):
            rect = QGraphicsRectItem(
                0, i * TRACK_HEIGHT, 10000, TRACK_HEIGHT)
            rect.setBrush(QBrush(QColor(
                COLORS["bg_alt"] if i % 2 == 0 else COLORS["bg"])))
            rect.setPen(QPen(Qt.PenStyle.NoPen))
            rect.setZValue(-10)
            self.addItem(rect)

        for i in range(num_tracks + 1):
            line = QGraphicsLineItem(
                0, i * TRACK_HEIGHT, 10000, i * TRACK_HEIGHT)
            line.setPen(QPen(QColor(COLORS["border"]), 1))
            line.setZValue(-5)
            self.addItem(line)

        self.playhead = QGraphicsLineItem(
            8, 0, 8, num_tracks * TRACK_HEIGHT)
        self.playhead.setPen(QPen(QColor("#ff4444"), 2))
        self.playhead.setZValue(100)
        self.addItem(self.playhead)

        self.clips = []

    def add_clip(self, stem_name, duration, track_index):
        c = ClipItem(stem_name, duration, track_index)
        self.addItem(c)
        self.clips.append(c)
        return c

    def set_playhead(self, seconds):
        self.playhead.setX(8 + seconds * PIXELS_PER_SECOND)

    def clear_clips(self):
        for c in self.clips:
            self.removeItem(c)
        self.clips.clear()


class TimelineView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(
            QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setStyleSheet(
            "QGraphicsView { background-color: " + COLORS["bg"] +
            "; border: 1px solid " + COLORS["border"] +
            "; border-radius: 8px; }")

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, 1.0)
            event.accept()
        else:
            super().wheelEvent(event)
FILE_END

echo ">> ui/channel_strip.py"
cat > "$ROOT/ui/channel_strip.py" <<'FILE_END'
"""Per-track channel strip: EQ, comp, reverb controls."""
from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider)
from PyQt6.QtCore import Qt

from .style import COLORS


class ChannelStrip(QFrame):
    def __init__(self, stem_name, dsp, parent=None):
        super().__init__(parent)
        color = COLORS["track_colors"].get(
            stem_name, COLORS["accent"])
        self.setStyleSheet(
            "QFrame { background-color: " + COLORS["bg_card"] +
            "; border-radius: 8px; border-left: 3px solid " +
            color + "; }")
        self.setFixedHeight(120)
        self.dsp = dsp

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        title = QLabel(stem_name.capitalize())
        title.setStyleSheet(
            "font-weight: 700; color: " + color +
            "; font-size: 13px;")
        layout.addWidget(title)

        eq_row = QHBoxLayout()
        eq_row.setSpacing(6)
        self.bass_slider = self._make_slider(
            -12, 12, int(dsp.eq_bass),
            lambda v: setattr(dsp, "eq_bass", float(v)))
        self.mid_slider = self._make_slider(
            -12, 12, int(dsp.eq_mid),
            lambda v: setattr(dsp, "eq_mid", float(v)))
        self.treble_slider = self._make_slider(
            -12, 12, int(dsp.eq_treble),
            lambda v: setattr(dsp, "eq_treble", float(v)))
        for label, slider in [("Bass", self.bass_slider),
                              ("Mid", self.mid_slider),
                              ("Treble", self.treble_slider)]:
            eq_row.addWidget(QLabel(label))
            eq_row.addWidget(slider, 1)
        layout.addLayout(eq_row)

        dyn_row = QHBoxLayout()
        dyn_row.setSpacing(6)
        self.comp_slider = self._make_slider(
            -40, 0, int(dsp.comp_threshold),
            lambda v: setattr(dsp, "comp_threshold", float(v)))
        self.reverb_slider = self._make_slider(
            0, 100, int(dsp.reverb_mix * 100),
            lambda v: setattr(dsp, "reverb_mix", v / 100.0))
        for label, slider in [("Comp", self.comp_slider),
                              ("Reverb", self.reverb_slider)]:
            dyn_row.addWidget(QLabel(label))
            dyn_row.addWidget(slider, 1)
        layout.addLayout(dyn_row)

    def _make_slider(self, lo, hi, init, setter):
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(init)

        def on_change(v):
            setter(v)
            self.dsp._filters_dirty = True

        s.valueChanged.connect(on_change)
        return s
FILE_END

echo ">> ui/workers.py"
cat > "$ROOT/ui/workers.py" <<'FILE_END'
"""QThread workers."""
import os
import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from core.separation import separate_file
from core.export import render_mix_to_wav, encode_with_ffmpeg


class SeparationWorker(QThread):
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, input_file, output_dir, model_name,
                 providers="auto", precision="fp32", chunked=False):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.model_name = model_name
        self.providers = providers
        self.precision = precision
        self.chunked = chunked

    def run(self):
        try:
            def cb(pct, msg):
                self.progress.emit(pct, msg)
            result = separate_file(
                self.input_file, self.output_dir, self.model_name,
                providers=self.providers, precision=self.precision,
                chunked=self.chunked, progress_cb=cb)
            self.finished_ok.emit(result)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


class ExportWorker(QThread):
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, tracks, dsp_chains, output_path,
                 fmt_key, bitrate):
        super().__init__()
        self.tracks = tracks
        self.dsp_chains = dsp_chains
        self.output_path = output_path
        self.fmt_key = fmt_key
        self.bitrate = bitrate

    def run(self):
        tmp_wav = self.output_path + ".tmp.wav"
        try:
            def p1(p):
                self.progress.emit(int(p * 0.6))
            ok = render_mix_to_wav(
                self.tracks, self.dsp_chains, tmp_wav,
                progress_cb=p1)
            if not ok:
                self.failed.emit("No active tracks to export.")
                return

            def p2(p):
                self.progress.emit(60 + int(p * 0.4))
            ok = encode_with_ffmpeg(
                tmp_wav, self.output_path, self.fmt_key,
                self.bitrate, progress_cb=p2)
            if not ok:
                self.failed.emit("FFmpeg encoding failed.")
                return
            self.finished_ok.emit(self.output_path)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))
        finally:
            if os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass
FILE_END

# ===========================================================================
# APPS PACKAGE
# ===========================================================================

echo ">> apps/__init__.py"
cat > "$ROOT/apps/__init__.py" <<'FILE_END'
"""Entry points for StemsMix applications."""
FILE_END

echo ">> apps/mixer_app.py"
cat > "$ROOT/apps/mixer_app.py" <<'FILE_END'
"""StemsMix Mixer - simple slider mixer."""
import sys
import os
import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QPushButton,
                             QSlider, QLabel, QFileDialog, QGroupBox,
                             QCheckBox, QProgressBar)
from PyQt6.QtCore import Qt, QTimer

from ui.style import COLORS, QSS
from ui.dialogs import (PrerequisiteDialog, ModelSelectionDialog,
                        ExportDialog)
from ui.workers import SeparationWorker, ExportWorker
from core.resources import (run_prerequisite_checks,
                            check_system_resources)
from core.export import EXPORT_FORMATS
from core.audio_io import load_audio


STEM_NAMES = ('vocals', 'drums', 'bass', 'other', 'guitar', 'piano')


class StemTrack:
    def __init__(self, filepath, name):
        self.name = name
        self.filepath = filepath
        self.data, self.samplerate = load_audio(filepath)
        self.volume = 1.0
        self.muted = False
        self.position = 0

    def reset(self):
        self.position = 0


class AudioMixer:
    def __init__(self, tracks):
        self.tracks = tracks
        self.stream = None
        self.samplerate = tracks[0].samplerate if tracks else 44100
        self.is_playing = False
        self._total_frames = max(
            (len(t.data) for t in tracks), default=0)

    def start(self):
        if self.stream is not None:
            self.stop()
        self.stream = sd.OutputStream(
            samplerate=self.samplerate, channels=2,
            callback=self._callback, blocksize=1024)
        self.stream.start()
        self.is_playing = True

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False

    def current_position_seconds(self):
        if not self.tracks:
            return 0.0
        return self.tracks[0].position / self.samplerate

    def total_duration_seconds(self):
        if not self.samplerate:
            return 0.0
        return self._total_frames / self.samplerate

    def _callback(self, outdata, frames, time, status):
        if status:
            return
        mix = np.zeros((frames, 2), dtype='float32')
        for track in self.tracks:
            if track.muted:
                continue
            remaining = len(track.data) - track.position
            read = min(frames, remaining)
            if read > 0:
                chunk = track.data[
                    track.position:track.position + read]
                mix[:read] += chunk * track.volume
                track.position += read
                if track.position >= len(track.data):
                    track.position = 0
        outdata[:] = np.clip(mix, -1.0, 1.0)


class TrackRow(QWidget):
    def __init__(self, track, index, on_mute, on_volume,
                 parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "QWidget { background-color: " + COLORS["bg_card"] +
            "; border-radius: 8px; }")
        self.setFixedHeight(52)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(12)

        color = COLORS["track_colors"].get(
            track.name, COLORS["accent"])
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(
            "background-color: " + color + "; border-radius: 5px;")
        layout.addWidget(dot)

        name = QLabel(track.name.capitalize())
        name.setFixedWidth(80)
        name.setStyleSheet(
            "font-weight: 600; font-size: 13px; color: " +
            COLORS["text"] + ";")
        layout.addWidget(name)

        mute_cb = QCheckBox("Mute")
        mute_cb.setChecked(track.muted)
        mute_cb.toggled.connect(
            lambda checked: on_mute(index, checked))
        layout.addWidget(mute_cb)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 200)
        slider.setValue(int(track.volume * 100))
        slider.valueChanged.connect(
            lambda v: on_volume(index, v / 100.0))
        layout.addWidget(slider, 1)

        vol = QLabel("{}%".format(int(track.volume * 100)))
        vol.setFixedWidth(44)
        vol.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignVCenter)
        vol.setStyleSheet(
            "color: " + COLORS["text_dim"] +
            "; font-size: 12px;")
        slider.valueChanged.connect(
            lambda v: vol.setText("{}%".format(v)))
        layout.addWidget(vol)


class MixerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StemsMix")
        self.setGeometry(100, 100, 1040, 820)
        self.tracks = []
        self.mixer = None
        self.worker = None
        self.export_worker = None
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(200)
        self.playback_timer.timeout.connect(
            self.update_playback_position)
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(16)

        header = QWidget()
        header.setStyleSheet(
            "QWidget { background-color: " + COLORS["bg_alt"] +
            "; border-radius: 10px; }")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 16, 20, 16)
        hl.setSpacing(12)
        title = QLabel("StemsMix")
        title.setObjectName("header_title")
        hl.addWidget(title)
        sub = QLabel("Stem separation and mixing")
        sub.setObjectName("header_subtitle")
        hl.addWidget(sub)
        hl.addStretch()
        main.addWidget(header)

        top = QHBoxLayout()
        top.setSpacing(8)
        self.btn_open = QPushButton("Open Audio File...")
        self.btn_open.clicked.connect(self.open_audio_file)
        top.addWidget(self.btn_open)
        self.btn_load = QPushButton("Load Stems Folder")
        self.btn_load.clicked.connect(self.load_stems_folder)
        top.addWidget(self.btn_load)
        top.addStretch()
        self.btn_play = QPushButton("Play")
        self.btn_play.setObjectName("primary")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.toggle_playback)
        top.addWidget(self.btn_play)
        self.btn_stop = QPushButton("Stop / Reset")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_playback)
        top.addWidget(self.btn_stop)
        self.btn_export = QPushButton("Export Mix")
        self.btn_export.setObjectName("primary")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_mix_dialog)
        top.addWidget(self.btn_export)
        main.addLayout(top)

        self.progress_label = QLabel("")
        self.progress_label.setObjectName("progress_label")
        self.progress_label.setVisible(False)
        main.addWidget(self.progress_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        main.addWidget(self.progress)

        playback = QHBoxLayout()
        playback.setSpacing(10)
        self.position_label = QLabel("0:00 / 0:00")
        self.position_label.setObjectName("position_label")
        self.position_label.setFixedWidth(110)
        playback.addWidget(self.position_label)
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.sliderMoved.connect(
            self.seek_playback)
        playback.addWidget(self.position_slider, 1)
        main.addLayout(playback)

        self.tracks_group = QGroupBox("Tracks")
        self.tracks_layout = QVBoxLayout()
        self.tracks_layout.setSpacing(6)
        self.tracks_layout.setContentsMargins(10, 8, 10, 10)
        self.tracks_group.setLayout(self.tracks_layout)
        main.addWidget(self.tracks_group, 1)

        self.status_label = QLabel(
            "Open an audio file to separate, or load a folder of "
            "existing stems.")
        self.status_label.setObjectName("status_bar")
        self.status_label.setWordWrap(True)
        main.addWidget(self.status_label)

    def open_audio_file(self):
        if self.worker is not None and self.worker.isRunning():
            self.status_label.setText(
                "Separation already running.")
            return

        resources = check_system_resources()
        input_file, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", os.path.expanduser("~"),
            "Audio Files (*.mp3 *.wav *.m4a *.mp4 *.m4b *.aac "
            "*.flac *.ogg *.opus *.wma);;All Files (*)")
        if not input_file:
            self.status_label.setText("Cancelled.")
            return

        dlg = ModelSelectionDialog(resources, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self.status_label.setText("Cancelled.")
            return

        model = dlg.selected_model()
        providers = ["cpu"] if dlg.force_cpu() else "auto"
        chunked = dlg.use_chunked()

        default_out = os.path.splitext(input_file)[0] + "_stems"
        out_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose output folder (default: " + default_out + ")",
            os.path.dirname(input_file))
        if not out_dir:
            self.status_label.setText("Cancelled.")
            return

        self._set_running(True)
        self.progress.setValue(0)
        self.progress_label.setText("Starting separation...")
        self.status_label.setText(
            "Separating with " + model + "...")

        self.worker = SeparationWorker(
            input_file, out_dir, model,
            providers=providers, chunked=chunked)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(
            lambda r: self.on_done(r, out_dir))
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(
            lambda: self._set_running(False))
        self.worker.start()

    def on_progress(self, pct, msg):
        self.progress.setValue(pct)
        self.progress_label.setText(msg)

    def _set_running(self, running):
        self.progress.setVisible(running)
        self.progress_label.setVisible(running)
        self.btn_open.setEnabled(not running)
        self.btn_load.setEnabled(not running)

    def on_done(self, result, out_dir):
        self._load_folder(out_dir)
        self.status_label.setText(
            "Separation complete. Loaded {} stems.".format(
                len(self.tracks)))

    def on_failed(self, msg):
        self.status_label.setText(
            "Separation failed: " + msg)
        self.progress_label.setText("Failed.")

    def load_stems_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Stems Folder")
        if not folder:
            return
        self._load_folder(folder)

    def _load_folder(self, folder):
        loaded = []
        for name in STEM_NAMES:
            p = os.path.join(folder, name + ".wav")
            if os.path.exists(p):
                try:
                    loaded.append(StemTrack(p, name))
                except Exception as e:
                    print("Failed {}: {}".format(name, e))
        if not loaded:
            self.status_label.setText(
                "No valid stems found.")
            return
        self.tracks = loaded
        self._rebuild_tracks()
        if self.mixer:
            self.mixer.stop()
        self.mixer = AudioMixer(self.tracks)
        duration = self.mixer.total_duration_seconds()
        self.position_slider.blockSignals(True)
        self.position_slider.setRange(
            0, max(1, int(duration * 1000)))
        self.position_slider.setValue(0)
        self.position_slider.blockSignals(False)
        self.update_pos_label(0, duration)
        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_play.setText("Play")
        self.status_label.setText(
            "Loaded {} stems from {}".format(
                len(self.tracks), folder))

    def _rebuild_tracks(self):
        while self.tracks_layout.count():
            item = self.tracks_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, t in enumerate(self.tracks):
            self.tracks_layout.addWidget(
                TrackRow(t, i, self.set_mute, self.set_volume))

    def set_mute(self, i, m):
        if 0 <= i < len(self.tracks):
            self.tracks[i].muted = m

    def set_volume(self, i, v):
        if 0 <= i < len(self.tracks):
            self.tracks[i].volume = v

    def toggle_playback(self):
        if not self.mixer:
            return
        if self.mixer.is_playing:
            self.mixer.stop()
            self.playback_timer.stop()
            self.btn_play.setText("Play")
        else:
            for t in self.tracks:
                t.reset()
            self.mixer.start()
            self.playback_timer.start()
            self.btn_play.setText("Pause")

    def stop_playback(self):
        if self.mixer:
            self.mixer.stop()
        self.playback_timer.stop()
        for t in self.tracks:
            t.reset()
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(0)
        self.position_slider.blockSignals(False)
        total = (self.mixer.total_duration_seconds()
                 if self.mixer else 0)
        self.update_pos_label(0, total)
        self.btn_play.setText("Play")
        self.status_label.setText("Stopped.")

    def update_playback_position(self):
        if not self.mixer or not self.mixer.is_playing:
            return
        pos = self.mixer.current_position_seconds()
        total = self.mixer.total_duration_seconds()
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(int(pos * 1000))
        self.position_slider.blockSignals(False)
        self.update_pos_label(pos, total)

    def seek_playback(self, value):
        if not self.mixer or not self.tracks:
            return
        secs = value / 1000.0
        frame = int(secs * self.mixer.samplerate)
        for t in self.tracks:
            t.position = min(frame, len(t.data))
        self.update_pos_label(
            secs, self.mixer.total_duration_seconds())

    def update_pos_label(self, pos, total):
        self.position_label.setText(
            "{} / {}".format(self._fmt(pos), self._fmt(total)))

    @staticmethod
    def _fmt(s):
        return "{}:{:02d}".format(int(s // 60), int(s % 60))

    def export_mix_dialog(self):
        if not self.tracks:
            return
        dlg = ExportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self.status_label.setText("Export cancelled.")
            return
        fmt = dlg.selected_format()
        br = dlg.selected_bitrate()
        ext = EXPORT_FORMATS[fmt]["ext"]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mix As", "mix." + ext,
            EXPORT_FORMATS[fmt]["filter"])
        if not path:
            self.status_label.setText("Export cancelled.")
            return
        if not path.lower().endswith("." + ext):
            path += "." + ext

        self.progress.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Rendering mix...")
        self.progress.setValue(0)

        from core.dsp import DSPChain
        neutral_chains = [DSPChain(self.tracks[0].samplerate)
                          for _ in self.tracks]
        self.export_worker = ExportWorker(
            self.tracks, neutral_chains, path, fmt, br)
        self.export_worker.progress.connect(
            self.progress.setValue)
        self.export_worker.finished_ok.connect(
            self.on_export_ok)
        self.export_worker.failed.connect(self.on_export_fail)
        self.export_worker.start()

    def on_export_ok(self, path):
        self.progress.setVisible(False)
        self.progress_label.setVisible(False)
        self.status_label.setText("Exported to: " + path)

    def on_export_fail(self, msg):
        self.progress.setVisible(False)
        self.progress_label.setVisible(False)
        self.status_label.setText("Export failed: " + msg)

    def closeEvent(self, event):
        if self.mixer:
            self.mixer.stop()
        self.playback_timer.stop()
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(2000)
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.terminate()
            self.export_worker.wait(2000)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    results = run_prerequisite_checks()
    if any(r.status in ("warn", "fail") for r in results):
        dlg = PrerequisiteDialog(results)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)
    w = MixerWindow()
    w.show()
    sys.exit(app.exec())
FILE_END

echo ">> apps/daw.py"
cat > "$ROOT/apps/daw.py" <<'FILE_END'
"""StemsMix DAW - timeline, effects, batch."""
import sys
import os
import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QSplitter)
from PyQt6.QtCore import Qt, QTimer

from ui.style import COLORS, QSS
from ui.dialogs import PrerequisiteDialog, ExportDialog
from ui.workers import ExportWorker
from ui.timeline import TimelineScene, TimelineView
from ui.channel_strip import ChannelStrip
from core.resources import run_prerequisite_checks
from core.dsp import DSPChain
from core.export import EXPORT_FORMATS
from core.audio_io import load_audio


STEM_NAMES = ('vocals', 'drums', 'bass', 'other', 'guitar', 'piano')


class StemTrack:
    def __init__(self, filepath, name):
        self.name = name
        self.filepath = filepath
        self.data, self.samplerate = load_audio(filepath)
        self.volume = 1.0
        self.muted = False
        self.position = 0

    def reset(self):
        self.position = 0


class TimelineMixer:
    def __init__(self, tracks, dsp_chains):
        self.tracks = tracks
        self.dsp_chains = dsp_chains
        self.samplerate = tracks[0].samplerate if tracks else 44100
        self.stream = None
        self.is_playing = False
        self._total_frames = max(
            (len(t.data) for t in tracks), default=0)

    def start(self):
        if self.stream is not None:
            self.stop()
        self.stream = sd.OutputStream(
            samplerate=self.samplerate, channels=2,
            callback=self._callback, blocksize=512)
        self.stream.start()
        self.is_playing = True

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False

    def current_position_seconds(self):
        if not self.tracks:
            return 0.0
        return self.tracks[0].position / self.samplerate

    def _callback(self, outdata, frames, time, status):
        if status:
            return
        mix = np.zeros((frames, 2), dtype='float32')
        for track, dsp in zip(self.tracks, self.dsp_chains):
            if track.muted:
                continue
            remaining = len(track.data) - track.position
            read = min(frames, remaining)
            if read <= 0:
                continue
            chunk = track.data[
                track.position:track.position + read].copy()
            chunk = dsp.process(chunk) * track.volume
            mix[:read] += chunk
            track.position += read
            if track.position >= len(track.data):
                track.position = 0
        outdata[:] = np.clip(mix, -1.0, 1.0)


class DAWWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StemsMix DAW")
        self.setGeometry(100, 100, 1400, 900)
        self.tracks = []
        self.dsp_chains = []
        self.mixer = None
        self.worker = None
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(40)
        self.playback_timer.timeout.connect(self.update_playhead)
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setContentsMargins(16, 16, 16, 16)
        main.setSpacing(12)

        header = QWidget()
        header.setStyleSheet(
            "QWidget { background-color: " + COLORS["bg_alt"] +
            "; border-radius: 10px; }")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 14, 20, 14)
        title = QLabel("StemsMix DAW")
        title.setObjectName("header_title")
        hl.addWidget(title)
        sub = QLabel("Timeline and Effects")
        sub.setObjectName("header_subtitle")
        hl.addWidget(sub)
        hl.addStretch()
        main.addWidget(header)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self.btn_load = QPushButton("Load Stems Folder...")
        self.btn_load.clicked.connect(self.load_stems)
        toolbar.addWidget(self.btn_load)
        toolbar.addStretch()
        self.btn_play = QPushButton("Play")
        self.btn_play.setObjectName("primary")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.toggle_playback)
        toolbar.addWidget(self.btn_play)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_playback)
        toolbar.addWidget(self.btn_stop)
        self.btn_export = QPushButton("Export Mix")
        self.btn_export.setObjectName("primary")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_mix)
        toolbar.addWidget(self.btn_export)
        main.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.scene = TimelineScene(num_tracks=6)
        self.view = TimelineView(self.scene)
        self.view.setMinimumHeight(300)
        splitter.addWidget(self.view)

        self.strips_container = QWidget()
        self.strips_layout = QHBoxLayout(self.strips_container)
        self.strips_layout.setContentsMargins(0, 0, 0, 0)
        self.strips_layout.setSpacing(8)
        splitter.addWidget(self.strips_container)
        splitter.setSizes([400, 200])
        main.addWidget(splitter, 1)

        self.status = QLabel(
            "Load a stems folder to begin.")
        self.status.setObjectName("status_bar")
        self.status.setWordWrap(True)
        main.addWidget(self.status)

    def load_stems(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Stems Folder")
        if not folder:
            return
        loaded = []
        for name in STEM_NAMES:
            p = os.path.join(folder, name + ".wav")
            if os.path.exists(p):
                try:
                    loaded.append(StemTrack(p, name))
                except Exception as e:
                    print("Failed {}: {}".format(name, e))
        if not loaded:
            self.status.setText("No valid stems found.")
            return
        self._install_tracks(loaded)
        self.status.setText(
            "Loaded {} stems from {}".format(
                len(loaded), folder))

    def _install_tracks(self, tracks):
        self.tracks = tracks
        self.dsp_chains = [DSPChain(t.samplerate)
                           for t in tracks]
        self.scene.clear_clips()
        self.scene.num_tracks = len(tracks)
        for i, t in enumerate(tracks):
            duration = len(t.data) / t.samplerate
            self.scene.add_clip(t.name, duration, i)

        while self.strips_layout.count():
            item = self.strips_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, t in enumerate(tracks):
            self.strips_layout.addWidget(
                ChannelStrip(t.name, self.dsp_chains[i]))

        if self.mixer:
            self.mixer.stop()
        self.mixer = TimelineMixer(tracks, self.dsp_chains)
        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(True)

    def toggle_playback(self):
        if not self.mixer:
            return
        if self.mixer.is_playing:
            self.mixer.stop()
            self.playback_timer.stop()
            self.btn_play.setText("Play")
        else:
            for t in self.tracks:
                t.reset()
            self.mixer.start()
            self.playback_timer.start()
            self.btn_play.setText("Pause")

    def stop_playback(self):
        if self.mixer:
            self.mixer.stop()
        self.playback_timer.stop()
        for t in self.tracks:
            t.reset()
        self.scene.set_playhead(0)
        self.btn_play.setText("Play")
        self.status.setText("Stopped.")

    def update_playhead(self):
        if self.mixer and self.mixer.is_playing:
            self.scene.set_playhead(
                self.mixer.current_position_seconds())

    def export_mix(self):
        if not self.tracks:
            return
        dlg = ExportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        fmt = dlg.selected_format()
        br = dlg.selected_bitrate()
        ext = EXPORT_FORMATS[fmt]["ext"]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mix As", "mix." + ext,
            EXPORT_FORMATS[fmt]["filter"])
        if not path:
            return
        if not path.lower().endswith("." + ext):
            path += "." + ext

        self.worker = ExportWorker(
            self.tracks, self.dsp_chains, path, fmt, br)

        def on_progress(p):
            self.status.setText(
                "Exporting... {}%".format(p))

        def on_ok(p):
            self.status.setText("Exported: " + p)

        def on_fail(e):
            self.status.setText("Failed: " + e)

        self.worker.progress.connect(on_progress)
        self.worker.finished_ok.connect(on_ok)
        self.worker.failed.connect(on_fail)
        self.worker.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    results = run_prerequisite_checks()
    if any(r.status in ("warn", "fail") for r in results):
        dlg = PrerequisiteDialog(results)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)
    w = DAWWindow()
    w.show()
    sys.exit(app.exec())
FILE_END

# ===========================================================================
# VERIFY, ZIP, CLEAN
# ===========================================================================

echo ""
echo ">> Verifying structure..."
TOTAL_FILES=$(find "$ROOT" -type f | wc -l)
echo "   Total files created: $TOTAL_FILES"
echo "   Expected:            16"

echo ""
echo ">> File tree:"
if command -v tree >/dev/null 2>&1; then
    tree "$ROOT"
else
    find "$ROOT" -type f | sort
fi

echo ""
echo ">> Creating $ZIP_NAME..."
zip -r "$ZIP_NAME" "$ROOT" -x '*.pyc' -x '*__pycache__*' > /dev/null

echo ">> Removing intermediate directory..."
rm -rf "$ROOT"

echo ""
echo "=================================================="
echo " Done. Created: $ZIP_NAME"
echo " Extract with:  unzip $ZIP_NAME"
echo " Run with:      cd $ROOT && python3.12 -m venv .venv"
echo "                source .venv/bin/activate"
echo "                pip install -r requirements.txt"
echo "                python apps/mixer_app.py"
echo "                python apps/daw.py"
echo "=================================================="