# mixer_app.py
import sys
import os
import subprocess
import traceback
import inspect
import re
from dataclasses import dataclass

import numpy as np
import soundfile as sf
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

_SEGMENT_SUPPORTED = False
if demucs_onnx is not None:
    try:
        _params = inspect.signature(demucs_onnx.separate).parameters
        _SEGMENT_SUPPORTED = "segment" in _params
    except (TypeError, ValueError):
        _SEGMENT_SUPPORTED = False

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QLabel,
                             QFileDialog, QGroupBox, QCheckBox, QProgressBar,
                             QDialog, QDialogButtonBox, QComboBox, QFormLayout,
                             QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer


# ---------------------------------------------------------------------------
# Prerequisite check system
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    blocking: bool


MIN_CPU_CORES_FAIL = 2
MIN_CPU_CORES_WARN = 4
MIN_RAM_GB_FAIL = 4
MIN_RAM_GB_WARN = 8
MIN_DISK_GB_FAIL = 1
MIN_DISK_GB_WARN = 5


def run_prerequisite_checks():
    results = []

    if SOUNDDEVICE_IMPORT_ERROR:
        results.append(CheckResult(
            "PortAudio", "fail",
            f"Audio playback unavailable: {SOUNDDEVICE_IMPORT_ERROR}. "
            f"Install with: sudo apt install portaudio19-dev", True))
    else:
        results.append(CheckResult("PortAudio", "pass", "Available", False))

    if DEMUCS_IMPORT_ERROR:
        results.append(CheckResult(
            "demucs-onnx", "fail",
            f"Separation engine unavailable: {DEMUCS_IMPORT_ERROR}. "
            f"Install with: pip install demucs-onnx", True))
    else:
        seg_note = " (segment parameter supported)" if _SEGMENT_SUPPORTED else ""
        results.append(CheckResult("demucs-onnx", "pass", "Available" + seg_note, False))

    ffmpeg_ok = False
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        ffmpeg_ok = (r.returncode == 0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        ffmpeg_ok = False

    if ffmpeg_ok:
        results.append(CheckResult("FFmpeg", "pass", "Found on PATH", False))
    else:
        results.append(CheckResult(
            "FFmpeg", "warn",
            "Not found — MP3, AAC and M4A export will not work. "
            "Install with: sudo apt install ffmpeg", False))

    try:
        cores = psutil.cpu_count(logical=True) or 1
    except Exception:
        cores = 1

    if cores >= MIN_CPU_CORES_WARN:
        results.append(CheckResult("CPU", "pass", f"{cores} logical cores", False))
    elif cores >= MIN_CPU_CORES_FAIL:
        results.append(CheckResult("CPU", "warn",
                                   f"Only {cores} cores — separation will be slow", False))
    else:
        results.append(CheckResult("CPU", "fail",
                                   f"Only {cores} core — at least {MIN_CPU_CORES_FAIL} required", True))

    try:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        avail_gb = mem.available / (1024 ** 3)
    except Exception:
        total_gb, avail_gb = 0.0, 0.0

    ram_msg = f"{total_gb:.1f} GB total, {avail_gb:.1f} GB available"
    if total_gb >= MIN_RAM_GB_WARN:
        results.append(CheckResult("RAM", "pass", ram_msg, False))
    elif total_gb >= MIN_RAM_GB_FAIL:
        results.append(CheckResult("RAM", "warn",
                                   f"{ram_msg} — 6-stem separation may freeze the system", False))
    else:
        results.append(CheckResult("RAM", "fail",
                                   f"{ram_msg} — at least {MIN_RAM_GB_FAIL} GB required", True))

    try:
        disk = psutil.disk_usage(os.path.expanduser("~"))
        free_gb = disk.free / (1024 ** 3)
    except Exception:
        free_gb = 0.0

    disk_msg = f"{free_gb:.1f} GB free in home directory"
    if free_gb >= MIN_DISK_GB_WARN:
        results.append(CheckResult("Disk space", "pass", disk_msg, False))
    elif free_gb >= MIN_DISK_GB_FAIL:
        results.append(CheckResult("Disk space", "warn",
                                   f"{disk_msg} — model download may fail", False))
    else:
        results.append(CheckResult("Disk space", "fail",
                                   f"{disk_msg} — at least {MIN_DISK_GB_FAIL} GB required", True))

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            results.append(CheckResult("GPU", "info", "CUDA available (GPU will be used)", False))
        elif "CoreMLExecutionProvider" in providers:
            results.append(CheckResult("GPU", "info", "CoreML available (Apple Silicon acceleration)", False))
        else:
            results.append(CheckResult("GPU", "info", "CPU only (no GPU acceleration detected)", False))
    except ImportError:
        results.append(CheckResult("GPU", "info", "onnxruntime not installed — detection skipped", False))

    return results


class PrerequisiteDialog(QDialog):
    STATUS_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌", "info": "ℹ️"}

    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results
        self.setWindowTitle("System Prerequisites")
        self.setMinimumWidth(620)

        self.has_blocking_failure = any(
            r.blocking and r.status == "fail" for r in results)

        layout = QVBoxLayout(self)

        header_text = (
            "<b>Cannot start the application.</b><br>"
            "One or more required components are missing. "
            "Fix the items marked ❌ before trying again."
            if self.has_blocking_failure else
            "<b>Some warnings detected.</b><br>"
            "The app should work, but some features may be limited or unstable. "
            "You can continue at your own risk."
        )
        header = QLabel(header_text)
        header.setWordWrap(True)
        header.setStyleSheet(
            "QLabel { padding: 10px; border-radius: 4px; "
            + ("background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;"
               if self.has_blocking_failure else
               "background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba;")
            + " }"
        )
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(340)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        for r in results:
            inner_layout.addWidget(self._make_row(r))
        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox()
        if self.has_blocking_failure:
            buttons.addButton("Quit", QDialogButtonBox.ButtonRole.RejectRole)
        else:
            continue_btn = buttons.addButton(
                "Continue anyway", QDialogButtonBox.ButtonRole.AcceptRole)
            continue_btn.setDefault(True)
            buttons.addButton("Quit", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _make_row(self, r):
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(4, 4, 4, 4)
        icon = QLabel(self.STATUS_ICON.get(r.status, "•"))
        icon.setFixedWidth(28)
        row.addWidget(icon)
        name = QLabel(f"<b>{r.name}</b>")
        name.setFixedWidth(110)
        row.addWidget(name)
        msg = QLabel(r.message)
        msg.setWordWrap(True)
        row.addWidget(msg, 1)
        bg = {"pass": "#d4edda", "warn": "#fff3cd",
              "fail": "#f8d7da", "info": "#e2e3e5"}.get(r.status, "transparent")
        row_widget.setStyleSheet(
            f"QWidget {{ background-color: {bg}; border-radius: 3px; }}"
            "QLabel { background-color: transparent; }")
        return row_widget


# ---------------------------------------------------------------------------
# Export format definitions
# ---------------------------------------------------------------------------

EXPORT_FORMATS = {
    "WAV (lossless)": {"ext": "wav", "filter": "WAV Files (*.wav)",
                       "bitrates": [("Lossless (24-bit PCM)", None)]},
    "MP3": {"ext": "mp3", "filter": "MP3 Files (*.mp3)",
            "bitrates": [("128 kbps", "128k"), ("192 kbps", "192k"),
                         ("256 kbps", "256k"), ("320 kbps", "320k")]},
    "M4A (AAC)": {"ext": "m4a", "filter": "M4A Files (*.m4a)",
                  "bitrates": [("128 kbps", "128k"), ("192 kbps", "192k"),
                               ("256 kbps", "256k")]},
    "AAC": {"ext": "aac", "filter": "AAC Files (*.aac)",
            "bitrates": [("128 kbps", "128k"), ("192 kbps", "192k"),
                         ("256 kbps", "256k")]},
}

QUALITY_NOTES = {
    ("WAV (lossless)", "Lossless (24-bit PCM)"):
        "Uncompressed 24-bit PCM. Bit-identical to the mix. Large files (≈10 MB/min).",
    ("MP3", "128 kbps"): "Small files, audible loss on cymbals and vocals.",
    ("MP3", "192 kbps"): "Good balance of size and quality.",
    ("MP3", "256 kbps"): "Near-transparent; hard to tell from the original.",
    ("MP3", "320 kbps"): "Maximum MP3 quality; transparent for most listeners.",
    ("M4A (AAC)", "128 kbps"): "Small and efficient. Slightly softer highs.",
    ("M4A (AAC)", "192 kbps"): "Very good quality. The AAC sweet spot.",
    ("M4A (AAC)", "256 kbps"): "Excellent quality; transparent for most content.",
    ("AAC", "128 kbps"): "Raw AAC stream. Small files.",
    ("AAC", "192 kbps"): "Raw AAC stream. Very good quality.",
    ("AAC", "256 kbps"): "Raw AAC stream. Excellent quality.",
}


class ExportWorker(QThread):
    """Runs FFmpeg encoding in the background with real progress."""
    progress = pyqtSignal(int)      # 0-100
    finished_ok = pyqtSignal(str)   # output path
    failed = pyqtSignal(str)

    def __init__(self, input_wav, output_path, fmt_key, bitrate):
        super().__init__()
        self.input_wav = input_wav
        self.output_path = output_path
        self.fmt_key = fmt_key
        self.bitrate = bitrate

    def run(self):
        try:
            if self.fmt_key == "WAV (lossless)":
                import shutil
                if self.input_wav != self.output_path:
                    shutil.copyfile(self.input_wav, self.output_path)
                self.progress.emit(100)
                self.finished_ok.emit(self.output_path)
                return

            if self.fmt_key == "MP3":
                codec_args = ["-codec:a", "libmp3lame", "-b:a", self.bitrate]
            elif self.fmt_key == "M4A (AAC)":
                codec_args = ["-codec:a", "aac", "-b:a", self.bitrate]
            elif self.fmt_key == "AAC":
                codec_args = ["-codec:a", "aac", "-b:a", self.bitrate, "-f", "adts"]
            else:
                self.failed.emit(f"Unknown format: {self.fmt_key}")
                return

            cmd = ["ffmpeg", "-y", "-i", self.input_wav,
                   "-progress", "pipe:1", "-nostats"] + codec_args + [self.output_path]

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # FFmpeg -progress emits "out_time_ms=N" lines; convert to percent
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("out_time_ms="):
                    try:
                        us = int(line.split("=")[1])
                        # Duration in seconds from soundfile
                        info = sf.info(self.input_wav)
                        total_us = info.frames / info.samplerate * 1_000_000
                        if total_us > 0:
                            pct = min(100, int(us / total_us * 100))
                            self.progress.emit(pct)
                    except (ValueError, ZeroDivisionError):
                        pass
                elif line == "progress=end":
                    self.progress.emit(100)

            proc.wait()
            if proc.returncode != 0:
                err = proc.stderr.read()
                self.failed.emit(f"FFmpeg error: {err[-400:]}")
                return

            self.progress.emit(100)
            self.finished_ok.emit(self.output_path)
        except FileNotFoundError:
            self.failed.emit("FFmpeg not found on PATH")
        except Exception as e:
            self.failed.emit(str(e))


# ---------------------------------------------------------------------------
# System resource check
# ---------------------------------------------------------------------------

MIN_RAM_6_STEM_GB = 6
MIN_CORES_6_STEM = 4


def check_system_resources():
    try:
        mem = psutil.virtual_memory()
        ram_gb = mem.available / (1024 ** 3)
        cores = psutil.cpu_count(logical=True) or 1
    except Exception:
        return {"ram_gb": 0.0, "cores": 0, "can_run_6_stem": False,
                "reason": "Could not determine system resources."}

    can_run = True
    reason = None
    if ram_gb < MIN_RAM_6_STEM_GB:
        can_run = False
        reason = (f"Only {ram_gb:.1f} GB RAM available; "
                  f"the 6-stem model needs at least {MIN_RAM_6_STEM_GB} GB free.")
    elif cores < MIN_CORES_6_STEM:
        can_run = False
        reason = (f"Only {cores} CPU core(s) detected; "
                  f"6-stem separation needs at least {MIN_CORES_6_STEM}.")

    return {"ram_gb": ram_gb, "cores": cores,
            "can_run_6_stem": can_run, "reason": reason}


# ---------------------------------------------------------------------------
# Model selection dialog
# ---------------------------------------------------------------------------

class ModelSelectionDialog(QDialog):
    def __init__(self, resources, parent=None):
        super().__init__(parent)
        self.resources = resources
        self.setWindowTitle("Choose Separation Model")
        self.setMinimumWidth(580)

        layout = QVBoxLayout(self)

        info = QLabel(
            "Choose the separation mode. <b>High Quality</b> gives the best "
            "results but is slower. <b>Fast</b> uses a lighter model that is "
            "roughly 30% faster with slightly lower separation quality. "
            "The <b>6-stem</b> option adds guitar and piano tracks at the cost "
            "of much higher memory use."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        sys_line = QLabel(
            f"Detected: <b>{resources['ram_gb']:.1f} GB</b> available RAM, "
            f"<b>{resources['cores']}</b> CPU core(s).")
        sys_line.setWordWrap(True)
        layout.addWidget(sys_line)

        self.combo = QComboBox()
        self.combo.addItem(
            "High Quality 4-stem (htdemucs_ft) — best quality, slower",
            ("htdemucs_ft", None))
        self.combo.addItem(
            "Fast 4-stem (htdemucs) — ~30% faster, slightly lower quality",
            ("htdemucs", None))
        if resources["can_run_6_stem"]:
            self.combo.addItem(
                "Fine 6-stem (htdemucs_6s) — adds guitar + piano",
                ("htdemucs_6s", None))
        self.combo.currentIndexChanged.connect(self.on_model_changed)
        layout.addWidget(self.combo)

        self.force_cpu_cb = QCheckBox("Force CPU (slower, but safer on low-VRAM GPUs)")
        layout.addWidget(self.force_cpu_cb)

        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet(
            "QLabel { background-color: #fff3cd; padding: 8px; "
            "border: 1px solid #ffeeba; border-radius: 4px; color: #856404; }")
        layout.addWidget(self.warning_label)

        if not resources["can_run_6_stem"]:
            blocked = QLabel(
                f"⚠ <b>6-stem mode is disabled on this machine.</b><br>"
                f"{resources['reason']}")
            blocked.setWordWrap(True)
            blocked.setStyleSheet(
                "QLabel { background-color: #f8d7da; padding: 8px; "
                "border: 1px solid #f5c6cb; border-radius: 4px; color: #721c24; }")
            layout.addWidget(blocked)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.on_model_changed()

    def on_model_changed(self):
        model_name, _ = self.combo.currentData()
        if model_name == "htdemucs_6s":
            self.force_cpu_cb.setChecked(True)
            self.warning_label.setText(
                "⚠ The 6-stem model is RAM-heavy. Force CPU is pre-selected for "
                "safety. Test with a short clip first and monitor memory usage.")
        elif model_name == "htdemucs":
            self.force_cpu_cb.setChecked(False)
            self.warning_label.setText(
                "Fast 4-stem model selected. Single-model architecture, "
                "roughly 30% faster than the quality bag.")
        else:
            self.force_cpu_cb.setChecked(False)
            self.warning_label.setText(
                "High Quality 4-stem model selected — best separation quality, "
                "but the slowest of the three options.")

    def selected_model(self):
        return self.combo.currentData()[0]

    def force_cpu(self):
        return self.force_cpu_cb.isChecked()


# ---------------------------------------------------------------------------
# Export dialog
# ---------------------------------------------------------------------------

class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Mix")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItems(EXPORT_FORMATS.keys())
        self.format_combo.currentTextChanged.connect(self.on_format_changed)
        form.addRow("Format:", self.format_combo)

        self.bitrate_combo = QComboBox()
        form.addRow("Bitrate:", self.bitrate_combo)
        layout.addLayout(form)

        self.quality_label = QLabel()
        self.quality_label.setWordWrap(True)
        self.quality_label.setMinimumHeight(70)
        self.quality_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; padding: 8px; "
            "border: 1px solid #ccc; border-radius: 4px; }")
        layout.addWidget(self.quality_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
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
        self.bitrate_combo.currentIndexChanged.connect(self.update_quality_label)
        self.update_quality_label()

    def update_quality_label(self):
        fmt = self.format_combo.currentText()
        br_label = self.bitrate_combo.currentText()
        self.quality_label.setText(QUALITY_NOTES.get((fmt, br_label), ""))

    def selected_format(self):
        return self.format_combo.currentText()

    def selected_bitrate(self):
        return self.bitrate_combo.currentData()


# ---------------------------------------------------------------------------
# Separation worker with chunked progress
# ---------------------------------------------------------------------------

class SeparationWorker(QThread):
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)   # percent, status message

    def __init__(self, input_file, output_dir, model_name,
                 providers="auto", precision="fp32", chunk_seconds=30):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.model_name = model_name
        self.providers = providers
        self.precision = precision
        self.chunk_seconds = chunk_seconds

    def run(self):
        try:
            self.progress.emit(0, "Reading audio file…")
            audio, sr = sf.read(self.input_file, dtype="float32", always_2d=True)
            total_frames = audio.shape[0]
            chunk_frames = int(self.chunk_seconds * sr)
            n_chunks = max(1, (total_frames + chunk_frames - 1) // chunk_frames)

            # Dict of stem name -> list of numpy chunks
            stem_chunks = {}
            tmp_dir = os.path.join(self.output_dir, "_tmp_chunks")
            os.makedirs(tmp_dir, exist_ok=True)

            for i in range(n_chunks):
                start = i * chunk_frames
                end = min(start + chunk_frames, total_frames)
                chunk = audio[start:end]

                # Write chunk to a temp WAV so demucs_onnx can read it
                chunk_path = os.path.join(tmp_dir, f"chunk_{i:04d}.wav")
                sf.write(chunk_path, chunk, sr)

                pct = int((i / n_chunks) * 100)
                self.progress.emit(pct, f"Separating chunk {i+1}/{n_chunks}…")

                kwargs = dict(
                    model=self.model_name,
                    providers=self.providers,
                    precision=self.precision,
                    progress=False,
                )
                # Run demucs on the chunk; results are written into tmp_dir
                result = demucs_onnx.separate(chunk_path, tmp_dir, **kwargs)

                # Load each stem result and append to accumulator
                for stem_name, stem_path in self._normalize_result(result).items():
                    data, _ = sf.read(stem_path, dtype="float32", always_2d=True)
                    stem_chunks.setdefault(stem_name, []).append(data)

                # Clean up the chunk input file
                try:
                    os.remove(chunk_path)
                except OSError:
                    pass

            # Concatenate and write final stems
            self.progress.emit(95, "Writing final stems…")
            for stem_name, chunks in stem_chunks.items():
                combined = np.concatenate(chunks, axis=0)
                out_path = os.path.join(self.output_dir, f"{stem_name}.wav")
                sf.write(out_path, combined, sr)

            # Clean up temp folder
            try:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

            self.progress.emit(100, "Done")
            self.finished_ok.emit({name: os.path.join(self.output_dir, f"{name}.wav")
                                   for name in stem_chunks.keys()})
        except Exception as e:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr)
            self.failed.emit(str(e))

    def _normalize_result(self, result):
        """Normalize whatever demucs_onnx.separate() returns into {name: path}."""
        if isinstance(result, dict):
            return {k: v for k, v in result.items()
                    if isinstance(v, str) and os.path.exists(v)}
        if isinstance(result, (list, tuple)):
            # Some versions return a list of paths
            out = {}
            for p in result:
                if isinstance(p, str) and os.path.exists(p):
                    name = os.path.splitext(os.path.basename(p))[0]
                    out[name] = p
            return out
        return {}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class StemTrack:
    def __init__(self, filepath, name):
        self.name = name
        self.filepath = filepath
        self.data, self.samplerate = sf.read(filepath, dtype='float32')
        if self.data.ndim == 1:
            self.data = np.column_stack((self.data, self.data))
        self.volume = 1.0
        self.muted = False
        self.position = 0

    def reset(self):
        self.position = 0


# ---------------------------------------------------------------------------
# Real-time mixer
# ---------------------------------------------------------------------------

class AudioMixer:
    def __init__(self, tracks):
        self.tracks = tracks
        self.stream = None
        self.samplerate = tracks[0].samplerate if tracks else 44100
        self.is_playing = False
        self._total_frames = max((len(t.data) for t in tracks), default=0)

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
        # Use the first track's position as the canonical playhead
        return self.tracks[0].position / self.samplerate

    def total_duration_seconds(self):
        if not self.samplerate:
            return 0.0
        return self._total_frames / self.samplerate

    def _callback(self, outdata, frames, time, status):
        if status:
            print(f"Stream status: {status}", file=sys.stderr)
        mix = np.zeros((frames, 2), dtype='float32')
        for track in self.tracks:
            if track.muted:
                continue
            remaining = len(track.data) - track.position
            read_frames = min(frames, remaining)
            if read_frames > 0:
                chunk = track.data[track.position:track.position + read_frames]
                mix[:read_frames] += chunk * track.volume
                track.position += read_frames
                if track.position >= len(track.data):
                    track.position = 0
        outdata[:] = np.clip(mix, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Render mix to temp WAV
# ---------------------------------------------------------------------------

def render_mix_to_wav(tracks, wav_path, samplerate=44100, progress_cb=None):
    active_tracks = [t for t in tracks if not t.muted]
    if not active_tracks:
        return False

    total_samples = max(len(t.data) for t in active_tracks)
    mix = np.zeros((total_samples, 2), dtype='float32')

    # Process in blocks so we can report progress
    block = 44100 * 5  # 5 seconds
    processed = 0
    for track in active_tracks:
        pos = 0
        while pos < len(track.data):
            end = min(pos + block, len(track.data))
            mix[pos:end] += track.data[pos:end] * track.volume
            pos = end
            processed += (end - pos)
            if progress_cb:
                # Rough progress across all tracks
                progress_cb(int(min(100, pos / len(track.data) * 50)))
    if mix.ndim == 1:
        mix = np.column_stack((mix, mix))
    mix = np.clip(mix, -1.0, 1.0)
    sf.write(wav_path, mix, samplerate, subtype='PCM_24')
    if progress_cb:
        progress_cb(100)
    return True


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

STEM_NAMES = ('vocals', 'drums', 'bass', 'other', 'guitar', 'piano')


class MixerWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python Stem Mixer - Moises Alternative")
        self.setGeometry(100, 100, 1000, 780)
        self.tracks = []
        self.mixer = None
        self.worker = None
        self.export_worker = None
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(200)  # 5 Hz
        self.playback_timer.timeout.connect(self.update_playback_position)
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        top_layout = QHBoxLayout()

        self.btn_open = QPushButton("Open Audio File…")
        self.btn_open.clicked.connect(self.open_audio_file)
        top_layout.addWidget(self.btn_open)

        self.btn_load = QPushButton("Load Stems Folder")
        self.btn_load.clicked.connect(self.load_stems_folder)
        top_layout.addWidget(self.btn_load)

        self.btn_play = QPushButton("Play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.toggle_playback)
        top_layout.addWidget(self.btn_play)

        self.btn_stop = QPushButton("Stop / Reset")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_playback)
        top_layout.addWidget(self.btn_stop)

        self.btn_export = QPushButton("Export Mix")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_mix_dialog)
        top_layout.addWidget(self.btn_export)

        main_layout.addLayout(top_layout)

        # --- Separation progress bar (visible only during separation) ---
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        main_layout.addWidget(self.progress_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)

        # --- Playback position bar (visible only when tracks are loaded) ---
        playback_layout = QHBoxLayout()
        self.position_label = QLabel("0:00 / 0:00")
        self.position_label.setFixedWidth(110)
        playback_layout.addWidget(self.position_label)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.setValue(0)
        self.position_slider.sliderMoved.connect(self.seek_playback)
        playback_layout.addWidget(self.position_slider)

        main_layout.addLayout(playback_layout)

        # --- Track controls ---
        self.tracks_group = QGroupBox("Tracks")
        self.tracks_layout = QVBoxLayout()
        self.tracks_group.setLayout(self.tracks_layout)
        main_layout.addWidget(self.tracks_group)

        self.status_label = QLabel(
            "Open an audio file to separate, or load a folder of existing stems.")
        main_layout.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Separation
    # ------------------------------------------------------------------

    def open_audio_file(self):
        if self.worker is not None and self.worker.isRunning():
            self.status_label.setText("Separation already running, please wait.")
            return

        resources = check_system_resources()
        if not resources["can_run_6_stem"]:
            self.status_label.setText(f"6-stem disabled: {resources['reason']}")

        input_file, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", os.path.expanduser("~"),
            "Audio Files (*.mp3 *.wav *.m4a *.flac *.ogg);;All Files (*)")
        if not input_file:
            self.status_label.setText("No file selected. Cancelled.")
            return

        model_dlg = ModelSelectionDialog(resources, self)
        if model_dlg.exec() != QDialog.DialogCode.Accepted:
            self.status_label.setText("Cancelled. Nothing was separated.")
            return

        model_name = model_dlg.selected_model()
        force_cpu = model_dlg.force_cpu()
        providers = ["cpu"] if force_cpu else "auto"

        default_out = os.path.splitext(input_file)[0] + "_stems"
        output_dir = QFileDialog.getExistingDirectory(
            self, f"Choose output folder (default: {default_out})",
            os.path.dirname(input_file))
        if not output_dir:
            self.status_label.setText("Cancelled. Nothing was separated.")
            return

        os.makedirs(output_dir, exist_ok=True)

        self._set_separation_running(True)
        self.progress.setValue(0)
        self.progress_label.setText("Starting separation…")
        self.status_label.setText(
            f"Separating '{os.path.basename(input_file)}' with {model_name}…")

        self.worker = SeparationWorker(
            input_file, output_dir, model_name,
            providers=providers, precision="fp32")
        self.worker.progress.connect(self.on_separation_progress)
        self.worker.finished_ok.connect(
            lambda paths: self.on_separation_done(paths, output_dir))
        self.worker.failed.connect(self.on_separation_failed)
        self.worker.finished.connect(lambda: self._set_separation_running(False))
        self.worker.start()

    def on_separation_progress(self, pct, message):
        self.progress.setValue(pct)
        self.progress_label.setText(message)

    def _set_separation_running(self, running):
        self.progress.setVisible(running)
        self.progress_label.setVisible(running)
        self.btn_open.setEnabled(not running)
        self.btn_load.setEnabled(not running)
        if not running:
            self.progress.setValue(0)

    def on_separation_done(self, result_dict, output_dir):
        self._load_stems_from_folder(output_dir)
        self.status_label.setText(
            f"Separation complete. Loaded {len(self.tracks)} stems from {output_dir}")

    def on_separation_failed(self, message):
        self.status_label.setText(f"Separation failed: {message}")
        self.progress_label.setText("Separation failed.")
        print(f"Separation failed: {message}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Loading stems
    # ------------------------------------------------------------------

    def load_stems_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Stems Folder")
        if not folder:
            self.status_label.setText("No folder selected. Cancelled.")
            return
        self._load_stems_from_folder(folder)

    def _load_stems_from_folder(self, folder):
        stem_files = {name: os.path.join(folder, f"{name}.wav") for name in STEM_NAMES}

        loaded = []
        for name, path in stem_files.items():
            if os.path.exists(path):
                try:
                    loaded.append(StemTrack(path, name))
                except Exception as e:
                    print(f"Failed to load {name}: {e}", file=sys.stderr)

        if not loaded:
            self.status_label.setText(
                "No valid stem files found (expected vocals.wav, drums.wav, etc.).")
            return

        self.tracks = loaded
        self.rebuild_track_controls()

        if self.mixer:
            self.mixer.stop()
        self.mixer = AudioMixer(self.tracks)

        # Update playback slider range to reflect track duration
        duration = self.mixer.total_duration_seconds()
        self.position_slider.blockSignals(True)
        self.position_slider.setRange(0, max(1, int(duration * 1000)))
        self.position_slider.setValue(0)
        self.position_slider.blockSignals(False)
        self.update_position_label(0.0, duration)

        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_play.setText("Play")
        self.status_label.setText(f"Loaded {len(self.tracks)} stems from {folder}")

    # ------------------------------------------------------------------
    # Track controls
    # ------------------------------------------------------------------

    def rebuild_track_controls(self):
        while self.tracks_layout.count():
            item = self.tracks_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, track in enumerate(self.tracks):
            row = QHBoxLayout()
            label = QLabel(track.name.capitalize())
            label.setFixedWidth(80)
            row.addWidget(label)

            mute_cb = QCheckBox("Mute")
            mute_cb.setChecked(track.muted)
            mute_cb.toggled.connect(lambda checked, idx=i: self.set_mute(idx, checked))
            row.addWidget(mute_cb)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 200)
            slider.setValue(int(track.volume * 100))
            slider.valueChanged.connect(lambda val, idx=i: self.set_volume(idx, val / 100.0))
            row.addWidget(slider)

            vol_label = QLabel(f"{int(track.volume * 100)}%")
            vol_label.setFixedWidth(40)
            slider.valueChanged.connect(lambda val, lbl=vol_label: lbl.setText(f"{val}%"))
            row.addWidget(vol_label)

            self.tracks_layout.addLayout(row)

    def set_mute(self, index, muted):
        if 0 <= index < len(self.tracks):
            self.tracks[index].muted = muted

    def set_volume(self, index, volume):
        if 0 <= index < len(self.tracks):
            self.tracks[index].volume = volume

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def toggle_playback(self):
        if not self.mixer:
            return
        if self.mixer.is_playing:
            self.mixer.stop()
            self.playback_timer.stop()
            self.btn_play.setText("Play")
        else:
            for track in self.tracks:
                track.reset()
            self.mixer.start()
            self.playback_timer.start()
            self.btn_play.setText("Pause")

    def stop_playback(self):
        if self.mixer:
            self.mixer.stop()
        self.playback_timer.stop()
        for track in self.tracks:
            track.reset()
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(0)
        self.position_slider.blockSignals(False)
        self.update_position_label(0.0, self.mixer.total_duration_seconds() if self.mixer else 0)
        self.btn_play.setText("Play")
        self.status_label.setText("Stopped and reset to beginning.")

    def update_playback_position(self):
        if not self.mixer or not self.mixer.is_playing:
            return
        pos = self.mixer.current_position_seconds()
        total = self.mixer.total_duration_seconds()
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(int(pos * 1000))
        self.position_slider.blockSignals(False)
        self.update_position_label(pos, total)

    def seek_playback(self, value):
        if not self.mixer or not self.tracks:
            return
        seconds = value / 1000.0
        frame = int(seconds * self.mixer.samplerate)
        for track in self.tracks:
            track.position = min(frame, len(track.data))
        self.update_position_label(seconds, self.mixer.total_duration_seconds())

    def update_position_label(self, pos, total):
        self.position_label.setText(f"{self._fmt(pos)} / {self._fmt(total)}")

    @staticmethod
    def _fmt(seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}:{s:02d}"

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_mix_dialog(self):
        if not self.tracks:
            return

        dlg = ExportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self.status_label.setText("Export cancelled.")
            return

        fmt_key = dlg.selected_format()
        bitrate = dlg.selected_bitrate()
        ext = EXPORT_FORMATS[fmt_key]["ext"]
        file_filter = EXPORT_FORMATS[fmt_key]["filter"]

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mix As", f"mix.{ext}", file_filter)
        if not path:
            self.status_label.setText("Export cancelled.")
            return

        if not path.lower().endswith(f".{ext}"):
            path += f".{ext}"

        # Render mix to temp WAV with progress
        tmp_wav = path + ".tmp.wav"
        self.progress.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Rendering mix…")
        self.progress.setValue(0)
        QApplication.processEvents()

        try:
            ok = render_mix_to_wav(
                self.tracks, tmp_wav,
                samplerate=self.tracks[0].samplerate,
                progress_cb=lambda p: (self.progress.setValue(p),
                                       QApplication.processEvents()))
            if not ok:
                self.status_label.setText("Export failed — no active tracks.")
                return

            # Encode with FFmpeg in a background thread
            self.progress_label.setText("Encoding…")
            self.progress.setValue(0)
            self.export_worker = ExportWorker(tmp_wav, path, fmt_key, bitrate)
            self.export_worker.progress.connect(self.progress.setValue)
            self.export_worker.finished_ok.connect(
                lambda out: self.on_export_done(out, tmp_wav))
            self.export_worker.failed.connect(
                lambda msg: self.on_export_failed(msg, tmp_wav))
            self.export_worker.start()
        except Exception as e:
            self.status_label.setText(f"Export failed: {e}")
            self.progress.setVisible(False)
            self.progress_label.setVisible(False)

    def on_export_done(self, output_path, tmp_wav):
        self.progress.setVisible(False)
        self.progress_label.setVisible(False)
        if os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass
        self.status_label.setText(f"Exported mix to: {output_path}")

    def on_export_failed(self, message, tmp_wav):
        self.progress.setVisible(False)
        self.progress_label.setVisible(False)
        if os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass
        self.status_label.setText(f"Export failed: {message}")

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self.mixer:
            self.mixer.stop()
        self.playback_timer.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(2000)
        if self.export_worker is not None and self.export_worker.isRunning():
            self.export_worker.terminate()
            self.export_worker.wait(2000)
        event.accept()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)

    results = run_prerequisite_checks()
    needs_dialog = any(r.status in ("warn", "fail") for r in results)

    if needs_dialog:
        dlg = PrerequisiteDialog(results)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            print("Startup cancelled by prerequisite check.", file=sys.stderr)
            sys.exit(0)

    window = MixerWindow()
    window.show()
    sys.exit(app.exec())
