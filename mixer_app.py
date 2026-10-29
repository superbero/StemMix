# mixer_app.py
import sys
import os
import subprocess
import traceback
import inspect
import shutil
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

from style import COLORS, QSS


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
        self.setMinimumWidth(640)

        self.has_blocking_failure = any(
            r.blocking and r.status == "fail" for r in results)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

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
            "QLabel { padding: 14px; border-radius: 8px; font-size: 13px; "
            + ("background-color: #3a1e22; color: #f8b6be; border: 1px solid #5a2a30;"
               if self.has_blocking_failure else
               "background-color: #3a3020; color: #f0d090; border: 1px solid #5a4a20;")
            + " }"
        )
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
            buttons.addButton("Quit", QDialogButtonBox.ButtonRole.RejectRole)
        else:
            continue_btn = buttons.addButton(
                "Continue anyway", QDialogButtonBox.ButtonRole.AcceptRole)
            continue_btn.setObjectName("primary")
            continue_btn.setDefault(True)
            buttons.addButton("Quit", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _make_row(self, r):
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(10)

        icon = QLabel(self.STATUS_ICON.get(r.status, "•"))
        icon.setFixedWidth(24)
        icon.setStyleSheet("font-size: 16px;")
        row.addWidget(icon)

        name = QLabel(f"<b>{r.name}</b>")
        name.setFixedWidth(110)
        name.setStyleSheet(f"color: {COLORS['text']};")
        row.addWidget(name)

        msg = QLabel(r.message)
        msg.setWordWrap(True)
        msg.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
        row.addWidget(msg, 1)

        bg = {
            "pass": "#1e3a24",
            "warn": "#3a3020",
            "fail": "#3a1e22",
            "info": "#2a2a30",
        }.get(r.status, COLORS["bg_card"])
        row_widget.setStyleSheet(
            f"QWidget {{ background-color: {bg}; border-radius: 6px; }}"
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


# ---------------------------------------------------------------------------
# Export worker with real FFmpeg progress
# ---------------------------------------------------------------------------

class ExportWorker(QThread):
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(str)
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
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

            try:
                info = sf.info(self.input_wav)
                total_us = info.frames / info.samplerate * 1_000_000
            except Exception:
                total_us = 0

            for line in proc.stdout:
                line = line.strip()
                if line.startswith("out_time_ms=") and total_us > 0:
                    try:
                        us = int(line.split("=", 1)[1])
                        pct = max(0, min(100, int(us / total_us * 100)))
                        self.progress.emit(pct)
                    except (ValueError, ZeroDivisionError):
                        pass
                elif line == "progress=end":
                    self.progress.emit(100)

            proc.wait()
            if proc.returncode != 0:
                self.failed.emit(f"FFmpeg exited with code {proc.returncode}")
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
        self.setMinimumWidth(620)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("Separation Model")
        title.setObjectName("header_title")
        layout.addWidget(title)

        info = QLabel(
            "<b>High Quality</b> gives the best results but is slower. "
            "<b>Fast</b> uses a lighter model that is roughly 30% faster with "
            "slightly lower separation quality. The <b>6-stem</b> option adds "
            "guitar and piano tracks at the cost of much higher memory use."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 13px;")
        layout.addWidget(info)

        sys_line = QLabel(
            f"Detected: <b>{resources['ram_gb']:.1f} GB</b> available RAM, "
            f"<b>{resources['cores']}</b> CPU core(s).")
        sys_line.setWordWrap(True)
        sys_line.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
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

        self.chunked_cb = QCheckBox(
            "Use chunked separation (slower, but lower peak memory)")
        self.chunked_cb.setChecked(resources["ram_gb"] < 8)
        layout.addWidget(self.chunked_cb)

        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet(
            "QLabel { background-color: #3a3020; color: #f0d090; "
            "padding: 12px; border: 1px solid #5a4a20; border-radius: 6px; "
            "font-size: 12px; }")
        layout.addWidget(self.warning_label)

        if not resources["can_run_6_stem"]:
            blocked = QLabel(
                f"⚠ <b>6-stem mode is disabled on this machine.</b><br>"
                f"{resources['reason']}")
            blocked.setWordWrap(True)
            blocked.setStyleSheet(
                "QLabel { background-color: #3a1e22; color: #f8b6be; "
                "padding: 12px; border: 1px solid #5a2a30; border-radius: 6px; "
                "font-size: 12px; }")
            layout.addWidget(blocked)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn:
            ok_btn.setObjectName("primary")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.on_model_changed()

    def on_model_changed(self):
        model_name, _ = self.combo.currentData()
        # GPU stays on for every model — never auto-force CPU.
        self.force_cpu_cb.setChecked(False)
        if model_name == "htdemucs_6s":
            self.warning_label.setText(
                "6-stem (guitar + piano): uses Metal/MLX on Apple Silicon. "
                "Only check Force CPU if separation runs out of memory.")
        elif model_name == "htdemucs":
            self.warning_label.setText(
                "Fast 4-stem model selected. Single-model architecture, "
                "roughly 30% faster than the quality bag.")
        else:
            self.warning_label.setText(
                "High Quality 4-stem (htdemucs_ft): ~4x slower, GPU stays ON. "
                "Do not enable Force CPU unless you hit memory limits.")

    def selected_model(self):
        return self.combo.currentData()[0]

    def force_cpu(self):
        return self.force_cpu_cb.isChecked()

    def use_chunked(self):
        return self.chunked_cb.isChecked()


# ---------------------------------------------------------------------------
# Export dialog
# ---------------------------------------------------------------------------

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
        self.format_combo.currentTextChanged.connect(self.on_format_changed)
        form.addRow("Format:", self.format_combo)

        self.bitrate_combo = QComboBox()
        form.addRow("Bitrate:", self.bitrate_combo)
        layout.addLayout(form)

        self.quality_label = QLabel()
        self.quality_label.setWordWrap(True)
        self.quality_label.setMinimumHeight(60)
        self.quality_label.setStyleSheet(
            f"QLabel {{ background-color: {COLORS['bg_card']}; "
            f"color: {COLORS['text_dim']}; padding: 12px; "
            f"border: 1px solid {COLORS['border']}; border-radius: 6px; "
            f"font-size: 12px; }}")
        layout.addWidget(self.quality_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
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
# Separation worker — single-pass or chunked
# ---------------------------------------------------------------------------

class SeparationWorker(QThread):
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, input_file, output_dir, model_name,
                 providers="auto", precision="fp32",
                 chunked=False, chunk_seconds=60):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.model_name = model_name
        self.providers = providers
        self.precision = precision
        self.chunked = chunked
        self.chunk_seconds = chunk_seconds

    def run(self):
        try:
            if self.chunked:
                self._run_chunked()
            else:
                self._run_single_pass()
        except Exception as e:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr)
            self.failed.emit(str(e))

    def _run_single_pass(self):
        self.progress.emit(5, "Loading model and separating…")
        kwargs = dict(
            model=self.model_name,
            providers=self.providers,
            precision=self.precision,
            progress=False,
        )
        self.progress.emit(15, "Running separation (this may take a while)…")
        result = demucs_onnx.separate(self.input_file, self.output_dir, **kwargs)
        self.progress.emit(95, "Finalizing…")
        normalized = self._normalize_result(result)
        self.progress.emit(100, "Done")
        self.finished_ok.emit(normalized)

    def _run_chunked(self):
        self.progress.emit(0, "Reading audio file…")
        audio, sr = sf.read(self.input_file, dtype="float32", always_2d=True)
        total_frames = audio.shape[0]
        chunk_frames = int(self.chunk_seconds * sr)
        n_chunks = max(1, (total_frames + chunk_frames - 1) // chunk_frames)

        stem_chunks = {}
        tmp_dir = os.path.join(self.output_dir, "_tmp_chunks")
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            for i in range(n_chunks):
                start = i * chunk_frames
                end = min(start + chunk_frames, total_frames)
                chunk = audio[start:end]

                chunk_path = os.path.join(tmp_dir, f"chunk_{i:04d}.wav")
                sf.write(chunk_path, chunk, sr)

                pct = int((i / n_chunks) * 90)
                self.progress.emit(pct, f"Separating chunk {i+1}/{n_chunks}…")

                kwargs = dict(
                    model=self.model_name,
                    providers=self.providers,
                    precision=self.precision,
                    progress=False,
                )
                result = demucs_onnx.separate(chunk_path, tmp_dir, **kwargs)

                for stem_name, stem_path in self._normalize_result(result).items():
                    data, _ = sf.read(stem_path, dtype="float32", always_2d=True)
                    stem_chunks.setdefault(stem_name, []).append(data)

                try:
                    os.remove(chunk_path)
                except OSError:
                    pass

            self.progress.emit(92, "Writing final stems…")
            out_paths = {}
            for stem_name, chunks in stem_chunks.items():
                combined = np.concatenate(chunks, axis=0)
                out_path = os.path.join(self.output_dir, f"{stem_name}.wav")
                sf.write(out_path, combined, sr)
                out_paths[stem_name] = out_path

            self.progress.emit(100, "Done")
            self.finished_ok.emit(out_paths)
        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    def _normalize_result(self, result):
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
# Track row widget
# ---------------------------------------------------------------------------

class TrackRow(QWidget):
    def __init__(self, track, index, on_mute, on_volume, parent=None):
        super().__init__(parent)
        self.setObjectName("trackRow")
        self.setStyleSheet(
            f"QWidget#trackRow {{ background-color: {COLORS['bg_card']}; "
            f"border-radius: 8px; }}")
        self.setFixedHeight(52)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(12)

        color = COLORS["track_colors"].get(track.name, COLORS["accent"])
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"background-color: {color}; border-radius: 5px;")
        layout.addWidget(dot)

        name = QLabel(track.name.capitalize())
        name.setFixedWidth(80)
        name.setStyleSheet(
            f"font-weight: 600; font-size: 13px; color: {COLORS['text']};")
        layout.addWidget(name)

        mute_cb = QCheckBox("Mute")
        mute_cb.setChecked(track.muted)
        mute_cb.toggled.connect(lambda checked: on_mute(index, checked))
        layout.addWidget(mute_cb)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 200)
        slider.setValue(int(track.volume * 100))
        slider.valueChanged.connect(lambda val: on_volume(index, val / 100.0))
        layout.addWidget(slider, 1)

        vol_label = QLabel(f"{int(track.volume * 100)}%")
        vol_label.setFixedWidth(44)
        vol_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        vol_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 12px; "
            f"font-family: Menlo, Consolas, monospace;")
        slider.valueChanged.connect(lambda val: vol_label.setText(f"{val}%"))
        layout.addWidget(vol_label)


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

    total_work = sum(len(t.data) for t in active_tracks)
    done_work = 0
    block = samplerate * 5

    for track in active_tracks:
        pos = 0
        while pos < len(track.data):
            end = min(pos + block, len(track.data))
            mix[pos:end] += track.data[pos:end] * track.volume
            done_work += (end - pos)
            pos = end
            if progress_cb and total_work > 0:
                progress_cb(int(done_work / total_work * 100))

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
        self.setWindowTitle("StemsMix")
        self.setGeometry(100, 100, 1040, 820)
        self.tracks = []
        self.mixer = None
        self.worker = None
        self.export_worker = None
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(200)
        self.playback_timer.timeout.connect(self.update_playback_position)
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # --- Header bar ---
        header = QWidget()
        header.setObjectName("header")
        header.setStyleSheet(
            f"QWidget#header {{ background-color: {COLORS['bg_alt']}; "
            f"border-radius: 10px; }}")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 16, 20, 16)
        header_layout.setSpacing(12)

        title = QLabel("StemsMix")
        title.setObjectName("header_title")
        header_layout.addWidget(title)

        subtitle = QLabel("Stem separation & mixing")
        subtitle.setObjectName("header_subtitle")
        header_layout.addWidget(subtitle)

        header_layout.addStretch()
        main_layout.addWidget(header)

        # --- Top controls ---
        top_layout = QHBoxLayout()
        top_layout.setSpacing(8)

        self.btn_open = QPushButton("Open Audio File…")
        self.btn_open.clicked.connect(self.open_audio_file)
        top_layout.addWidget(self.btn_open)

        self.btn_load = QPushButton("Load Stems Folder")
        self.btn_load.clicked.connect(self.load_stems_folder)
        top_layout.addWidget(self.btn_load)

        top_layout.addStretch()

        self.btn_play = QPushButton("Play")
        self.btn_play.setObjectName("primary")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.toggle_playback)
        top_layout.addWidget(self.btn_play)

        self.btn_stop = QPushButton("Stop / Reset")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_playback)
        top_layout.addWidget(self.btn_stop)

        self.btn_export = QPushButton("Export Mix")
        self.btn_export.setObjectName("primary")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_mix_dialog)
        top_layout.addWidget(self.btn_export)

        main_layout.addLayout(top_layout)

        # --- Progress ---
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("progress_label")
        self.progress_label.setVisible(False)
        main_layout.addWidget(self.progress_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)

        # --- Playback position ---
        playback_layout = QHBoxLayout()
        playback_layout.setSpacing(10)

        self.position_label = QLabel("0:00 / 0:00")
        self.position_label.setObjectName("position_label")
        self.position_label.setFixedWidth(110)
        playback_layout.addWidget(self.position_label)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.setValue(0)
        self.position_slider.sliderMoved.connect(self.seek_playback)
        playback_layout.addWidget(self.position_slider, 1)

        main_layout.addLayout(playback_layout)

        # --- Tracks panel ---
        self.tracks_group = QGroupBox("Tracks")
        self.tracks_layout = QVBoxLayout()
        self.tracks_layout.setSpacing(6)
        self.tracks_layout.setContentsMargins(10, 8, 10, 10)
        self.tracks_group.setLayout(self.tracks_layout)
        main_layout.addWidget(self.tracks_group, 1)

        # --- Status bar ---
        self.status_label = QLabel(
            "Open an audio file to separate, or load a folder of existing stems.")
        self.status_label.setObjectName("status_bar")
        self.status_label.setWordWrap(True)
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
        chunked = model_dlg.use_chunked()
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
        mode_note = " (chunked)" if chunked else ""
        self.status_label.setText(
            f"Separating '{os.path.basename(input_file)}' with {model_name}{mode_note}…")

        self.worker = SeparationWorker(
            input_file, output_dir, model_name,
            providers=providers, precision="fp32", chunked=chunked)
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
            row = TrackRow(track, i, self.set_mute, self.set_volume)
            self.tracks_layout.addWidget(row)

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
        self.update_position_label(
            0.0, self.mixer.total_duration_seconds() if self.mixer else 0)
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
                self.progress.setVisible(False)
                self.progress_label.setVisible(False)
                return

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
    app.setStyleSheet(QSS)

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