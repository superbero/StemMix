#!/usr/bin/env bash
# apply-all-mods.sh
# Applies every modification we've built in this session to an
# existing StemsMix-modular project:
#
#   1. core/timeline_model.py    (new)   - Project / Track / Clip
#   2. core/engine.py            (new)   - timeline audio engine
#   3. core/separation.py        (rewrite) - MLX-first, cached Separator
#   4. ui/timeline.py            (rewrite) - waveforms, headers, playhead
#   5. ui/workers.py             (patch)  - BackgroundExportWorker
#   6. ui/dialogs.py             (patch)  - speed hints
#   7. apps/daw.py               (rewrite) - full new DAW
#
# Safe to run multiple times. Every touched file is backed up.

set -euo pipefail

if [ ! -d "core" ] || [ ! -d "ui" ] || [ ! -d "apps" ]; then
    echo "ERROR: Run this from the StemsMix-modular/ project root."
    exit 1
fi

STAMP=$(date +%s)
echo ""
echo "=================================================="
echo " Applying all StemsMix modifications"
echo " Backup suffix: .bak.$STAMP"
echo "=================================================="

backup() {
    if [ -f "$1" ] && [ ! -f "$1.bak.$STAMP" ]; then
        cp "$1" "$1.bak.$STAMP"
        echo "   backup: $1"
    fi
}

# ---------------------------------------------------------------------------
# 1. core/timeline_model.py
# ---------------------------------------------------------------------------
echo ""
echo ">> [1/7] core/timeline_model.py"
backup "core/timeline_model.py"
cat > core/timeline_model.py <<'FILE_END'
"""Timeline project state: tracks, clips, playhead. No audio, no Qt."""
import copy


class Clip:
    """A piece of audio placed on a timeline."""

    def __init__(self, stem_name, filepath, audio_data,
                 samplerate, start_time=0.0):
        self.id = id(self)
        self.stem_name = stem_name
        self.filepath = filepath
        self.data = audio_data
        self.samplerate = samplerate
        self.start_time = start_time
        self.duration = len(audio_data) / samplerate
        self.stretch = 1.0
        self.pitch = 1.0
        self.gain = 1.0
        self.muted = False
        self.soloed = False
        self.color_hint = None

    def end_time(self):
        return self.start_time + self.duration / self.stretch

    def contains_time(self, seconds):
        return self.start_time <= seconds < self.end_time()


class Track:
    """A lane that holds clips. Each track has its own DSP chain."""

    def __init__(self, name, color_hint=None):
        self.name = name
        self.color_hint = color_hint
        self.clips = []
        self.muted = False
        self.soloed = False
        self.volume = 1.0

    def add_clip(self, clip):
        self.clips.append(clip)

    def remove_clip(self, clip):
        if clip in self.clips:
            self.clips.remove(clip)

    def clips_at(self, seconds):
        return [c for c in self.clips if c.contains_time(seconds)]


class Project:
    """Top-level container for all tracks and global state."""

    def __init__(self):
        self.tracks = []
        self.samplerate = 44100
        self.total_duration = 0.0

    def add_track(self, track):
        self.tracks.append(track)
        self._recompute_duration()

    def remove_track(self, track):
        if track in self.tracks:
            self.tracks.remove(track)
        self._recompute_duration()

    def _recompute_duration(self):
        end = 0.0
        for t in self.tracks:
            for c in t.clips:
                end = max(end, c.end_time())
        self.total_duration = end

    def any_soloed(self):
        return any(t.soloed or any(c.soloed for c in t.clips)
                   for t in self.tracks)

    def is_audible(self, track, clip=None):
        any_solo = self.any_soloed()
        if track.muted:
            return False
        track_soloed = track.soloed
        clip_soloed = clip.soloed if clip else False
        if any_solo and not (track_soloed or clip_soloed):
            return False
        if clip is not None and clip.muted:
            return False
        return True

    def snapshot(self):
        return copy.deepcopy(self)
FILE_END

# ---------------------------------------------------------------------------
# 2. core/engine.py
# ---------------------------------------------------------------------------
echo ""
echo ">> [2/7] core/engine.py"
backup "core/engine.py"
cat > core/engine.py <<'FILE_END'
"""Real timeline audio engine with per-clip offsets, stretch, pitch."""
import numpy as np
import sounddevice as sd


class TimelineEngine:
    """Reads from a Project at a global playhead position."""

    def __init__(self, project, dsp_chain_lookup):
        self.project = project
        self.dsp_lookup = dsp_chain_lookup
        self.samplerate = project.samplerate
        self.stream = None
        self.is_playing = False
        self.playhead_seconds = 0.0
        self._playhead_frames = 0

    def start(self, from_seconds=None):
        if self.stream is not None:
            self.stop()
        if from_seconds is not None:
            self.seek(from_seconds)
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

    def seek(self, seconds):
        self.playhead_seconds = max(0.0, float(seconds))
        self._playhead_frames = int(self.playhead_seconds * self.samplerate)

    def _callback(self, outdata, frames, time, status):
        if status:
            return
        out = np.zeros((frames, 2), dtype='float32')
        sr = self.samplerate
        start_sec = self._playhead_frames / sr
        end_sec = (self._playhead_frames + frames) / sr

        for track in self.project.tracks:
            dsp = self.dsp_lookup(track.name)
            for clip in track.clips:
                clip_start = clip.start_time
                clip_end = clip.end_time()
                if clip_end <= start_sec or clip_start >= end_sec:
                    continue
                if not self.project.is_audible(track, clip):
                    continue
                block_from = max(start_sec, clip_start)
                block_to = min(end_sec, clip_end)
                if block_to <= block_from:
                    continue
                out_a = int((block_from - start_sec) * sr)
                out_b = int((block_to - start_sec) * sr)
                n = out_b - out_a
                if n <= 0:
                    continue
                in_a_sec = (block_from - clip_start) * clip.stretch
                in_b_sec = (block_to - clip_start) * clip.stretch
                in_a = in_a_sec * sr
                in_b = in_b_sec * sr
                chunk = self._read_stretched(clip, in_a, in_b, n)
                if chunk is None:
                    continue
                if clip.pitch != 1.0:
                    chunk = self._apply_pitch(chunk, clip.pitch)
                if chunk.shape[0] > n:
                    chunk = chunk[:n]
                elif chunk.shape[0] < n:
                    pad = np.zeros((n - chunk.shape[0], 2),
                                   dtype='float32')
                    chunk = np.vstack([chunk, pad])
                chunk = chunk * (clip.gain * track.volume)
                chunk = dsp.process(chunk)
                out[out_a:out_b] += chunk

        self._playhead_frames += frames
        self.playhead_seconds = self._playhead_frames / sr

        if (self.project.total_duration > 0
                and self.playhead_seconds >=
                self.project.total_duration):
            outdata[:] = np.clip(out, -1, 1)
            self.is_playing = False
            return

        outdata[:] = np.clip(out, -1.0, 1.0)

    def _read_stretched(self, clip, in_a, in_b, n_out):
        src = clip.data
        if src is None or src.shape[0] == 0:
            return None
        if src.ndim == 1:
            src = np.column_stack((src, src))
        t = np.linspace(in_a, in_b, n_out, endpoint=False)
        i0 = np.floor(t).astype(int)
        i1 = i0 + 1
        frac = (t - i0).astype('float32')
        i0 = np.clip(i0, 0, src.shape[0] - 1)
        i1 = np.clip(i1, 0, src.shape[0] - 1)
        a = src[i0]
        b = src[i1]
        frac2 = frac[:, np.newaxis]
        return a * (1 - frac2) + b * frac2

    def _apply_pitch(self, chunk, pitch_factor):
        if abs(pitch_factor - 1.0) < 1e-4:
            return chunk
        n_out = chunk.shape[0]
        n_in = max(2, int(n_out / pitch_factor))
        src = chunk[:n_in] if chunk.shape[0] >= n_in else np.pad(
            chunk, ((0, n_in - chunk.shape[0]), (0, 0)))
        t = np.linspace(0, src.shape[0] - 1, n_out)
        i0 = np.floor(t).astype(int)
        i1 = np.clip(i0 + 1, 0, src.shape[0] - 1)
        frac = (t - i0).astype('float32')[:, np.newaxis]
        out = src[i0] * (1 - frac) + src[i1] * frac
        return out.astype('float32')

    def render_offline(self, output_path,
                       start_seconds=0.0, end_seconds=None):
        import soundfile as sf
        if end_seconds is None:
            end_seconds = self.project.total_duration
        total_frames = int((end_seconds - start_seconds) * self.samplerate)
        if total_frames <= 0:
            return False
        out = np.zeros((total_frames, 2), dtype='float32')
        for track in self.project.tracks:
            dsp = self.dsp_lookup(track.name)
            for clip in track.clips:
                if not self.project.is_audible(track, clip):
                    continue
                cs = clip.start_time
                ce = clip.end_time()
                if ce <= start_seconds or cs >= end_seconds:
                    continue
                block_from = max(start_seconds, cs)
                block_to = min(end_seconds, ce)
                n = int((block_to - block_from) * self.samplerate)
                if n <= 0:
                    continue
                in_a = (block_from - cs) * clip.stretch * self.samplerate
                in_b = (block_to - cs) * clip.stretch * self.samplerate
                chunk = self._read_stretched(clip, in_a, in_b, n)
                if chunk is None:
                    continue
                if clip.pitch != 1.0:
                    chunk = self._apply_pitch(chunk, clip.pitch)
                chunk = chunk * (clip.gain * track.volume)
                chunk = dsp.process_offline(chunk)
                out_a = int((block_from - start_seconds) * self.samplerate)
                out_b = out_a + n
                out[out_a:out_b] += chunk[:out_b - out_a]
        out = np.clip(out, -1, 1)
        sf.write(output_path, out, self.samplerate, subtype='PCM_24')
        return True
FILE_END

# ---------------------------------------------------------------------------
# 3. core/separation.py
# ---------------------------------------------------------------------------
echo ""
echo ">> [3/7] core/separation.py"
backup "core/separation.py"
cat > core/separation.py <<'FILE_END'
"""Separation engine. Prefers demucs-mlx on Apple Silicon."""
import os
import platform
import shutil

import numpy as np
import soundfile as sf

from .audio_io import convert_to_wav, needs_ffmpeg_conversion
from .resources import DEMUCS_IMPORT_ERROR

_MLX_SEPARATOR = None
_MLX_MODEL_NAME = None
try:
    if (platform.system() == "Darwin"
            and platform.machine() == "arm64"):
        from demucs_mlx import Separator as MLXSeparator
        MLX_AVAILABLE = True
    else:
        MLXSeparator = None
        MLX_AVAILABLE = False
except ImportError:
    MLXSeparator = None
    MLX_AVAILABLE = False

if DEMUCS_IMPORT_ERROR is None:
    import demucs_onnx
else:
    demucs_onnx = None


def separate_file(input_path, output_dir, model_name,
                  providers="auto", precision="fp32",
                  chunked=False, chunk_seconds=60, progress_cb=None):
    def emit(pct, msg):
        if progress_cb:
            progress_cb(int(pct), str(msg))

    force_cpu = providers in ("cpu", ["cpu"])
    use_mlx = MLX_AVAILABLE and not force_cpu

    if not use_mlx and DEMUCS_IMPORT_ERROR:
        raise RuntimeError(
            "No separation backend available. Install demucs-onnx "
            "or demucs-mlx.")

    converted = None
    input_to_use = input_path
    try:
        if needs_ffmpeg_conversion(input_path):
            emit(1, "Converting to WAV...")
            try:
                converted = convert_to_wav(input_path)
                input_to_use = converted
                emit(5, "Converted.")
            except RuntimeError as e:
                raise RuntimeError(str(e))

        os.makedirs(output_dir, exist_ok=True)

        if use_mlx:
            return _run_mlx(input_to_use, output_dir, model_name, emit)
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


def _run_mlx(input_path, output_dir, model_name, emit):
    global _MLX_SEPARATOR, _MLX_MODEL_NAME
    if _MLX_SEPARATOR is None or _MLX_MODEL_NAME != model_name:
        emit(10, "Loading MLX model (one-time compile)...")
        _MLX_SEPARATOR = MLXSeparator(model=model_name)
        _MLX_MODEL_NAME = model_name
    else:
        emit(10, "Reusing cached MLX model...")
    emit(20, "Separating with Metal GPU...")
    origin, stems = _MLX_SEPARATOR.separate_audio_file(input_path)
    emit(90, "Writing stems...")
    out_paths = {}
    sr = getattr(_MLX_SEPARATOR, "samplerate", 44100)
    for name, audio in stems.items():
        out = np.transpose(audio) if audio.ndim == 2 else audio
        out_path = os.path.join(output_dir, name + ".wav")
        sf.write(out_path, out, sr)
        out_paths[name] = out_path
    emit(100, "Done")
    return out_paths


def _run_single_pass(input_path, output_dir, model_name,
                     providers, precision, emit):
    emit(10, "Loading model and separating...")
    kwargs = dict(model=model_name, providers=providers,
                  precision=precision, progress=False)
    emit(20, "Running separation...")
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
            chunk_path = os.path.join(tmp_dir, "chunk_{:04d}.wav".format(i))
            sf.write(chunk_path, chunk, sr)
            pct = 10 + int((i / n_chunks) * 80)
            emit(pct, "Separating chunk {}/{}...".format(i + 1, n_chunks))
            kwargs = dict(model=model_name, providers=providers,
                          precision=precision, progress=False)
            result = demucs_onnx.separate(chunk_path, tmp_dir, **kwargs)
            for stem_name, stem_path in _normalize_result(result).items():
                data, _ = sf.read(stem_path, dtype="float32", always_2d=True)
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

# ---------------------------------------------------------------------------
# 4. ui/timeline.py
# ---------------------------------------------------------------------------
echo ""
echo ">> [4/7] ui/timeline.py"
backup "ui/timeline.py"
cat > ui/timeline.py <<'FILE_END'
"""Timeline scene/view/clip widgets with track headers and waveforms."""
import numpy as np
from PyQt6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsItem,
                             QGraphicsRectItem, QGraphicsLineItem,
                             QGraphicsTextItem, QGraphicsPathItem,
                             QWidget, QHBoxLayout, QVBoxLayout,
                             QPushButton, QLabel, QSlider)
from PyQt6.QtGui import (QBrush, QPen, QColor, QFont, QPainter,
                         QPainterPath, QWheelEvent)
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal

from .style import COLORS


TRACK_HEIGHT = 90
HEADER_WIDTH = 180
PIXELS_PER_SECOND = 80


class ClipItem(QGraphicsRectItem):
    def __init__(self, stem_name, audio_data, samplerate,
                 track_index, parent=None):
        duration = len(audio_data) / samplerate if samplerate else 0
        h = TRACK_HEIGHT - 16
        w = max(20, duration * PIXELS_PER_SECOND)
        super().__init__(0, 0, w, h, parent)
        self.stem_name = stem_name
        self.track_index = track_index
        self.duration = duration
        self.clip_ref = None

        color = QColor(COLORS["track_colors"].get(
            stem_name, COLORS["accent"]))
        self.base_color = color
        self.setBrush(QBrush(color.darker(180)))
        self.setPen(QPen(color, 2))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setY(track_index * TRACK_HEIGHT + 8)
        self.setX(HEADER_WIDTH + 8)

        self._waveform = QPainterPath()
        self._build_waveform(audio_data, w, h)
        wave_item = QGraphicsPathItem(self._waveform, self)
        wave_item.setPen(QPen(color.lighter(150), 1))
        wave_item.setZValue(1)

        label = QGraphicsTextItem(stem_name.capitalize(), self)
        label.setDefaultTextColor(QColor("#ffffff"))
        f = QFont()
        f.setBold(True)
        f.setPointSize(10)
        label.setFont(f)
        label.setPos(8, 4)
        label.setZValue(2)

        dur_text = "{}:{:02d}".format(
            int(duration // 60), int(duration % 60))
        dl = QGraphicsTextItem(dur_text, self)
        dl.setDefaultTextColor(QColor("#cccccc"))
        f2 = QFont()
        f2.setPointSize(8)
        dl.setFont(f2)
        dl.setPos(8, h - 18)
        dl.setZValue(2)

    def _build_waveform(self, audio, w, h):
        if audio is None or len(audio) == 0 or w < 4:
            return
        n_target = max(10, int(w * 2))
        if len(audio) > n_target:
            step = len(audio) // n_target
            samples = audio[::step][:n_target]
        else:
            samples = audio
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        center = h / 2
        scale = (h / 2) - 12
        self._waveform.moveTo(0, center)
        for i, s in enumerate(samples):
            x = i * w / len(samples)
            y = center - float(np.clip(s, -1, 1)) * scale
            self._waveform.lineTo(x, y)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            new_pos = value
            lane = max(0, round(new_pos.y() / TRACK_HEIGHT))
            return QPointF(max(HEADER_WIDTH + 2, new_pos.x()),
                           lane * TRACK_HEIGHT + 8)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.track_index = round(self.y() / TRACK_HEIGHT)
            if self.clip_ref is not None:
                self.clip_ref.start_time = max(
                    0.0, (self.x() - HEADER_WIDTH - 8) / PIXELS_PER_SECOND)
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        if self.isSelected():
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.rect())


class PlayheadItem(QGraphicsItem):
    def __init__(self, num_tracks, parent=None):
        super().__init__(parent)
        self.num_tracks = num_tracks
        self.x_pos = HEADER_WIDTH + 8
        self.setZValue(100)

    def boundingRect(self):
        return QRectF(self.x_pos - 6, 0, 12,
                      self.num_tracks * TRACK_HEIGHT + 20)

    def paint(self, painter, option, widget=None):
        painter.setPen(QPen(QColor("#ff4444"), 2))
        painter.drawLine(QPointF(self.x_pos, 0),
                         QPointF(self.x_pos,
                                 self.num_tracks * TRACK_HEIGHT))
        painter.setBrush(QBrush(QColor("#ff4444")))
        painter.setPen(Qt.PenStyle.NoPen)
        path = QPainterPath()
        path.moveTo(self.x_pos - 6, 0)
        path.lineTo(self.x_pos + 6, 0)
        path.lineTo(self.x_pos, 10)
        path.closeSubpath()
        painter.drawPath(path)

    def set_position(self, x):
        self.prepareGeometryChange()
        self.x_pos = x
        self.update()


class TimelineScene(QGraphicsScene):
    def __init__(self, num_tracks=6, parent=None):
        super().__init__(parent)
        self.num_tracks = num_tracks
        self.setBackgroundBrush(QBrush(QColor(COLORS["bg"])))
        self.setSceneRect(0, 0, 10000, num_tracks * TRACK_HEIGHT + 20)
        for i in range(num_tracks):
            rect = QGraphicsRectItem(HEADER_WIDTH, i * TRACK_HEIGHT,
                                     10000, TRACK_HEIGHT)
            rect.setBrush(QBrush(QColor(
                COLORS["bg_alt"] if i % 2 == 0 else COLORS["bg"])))
            rect.setPen(QPen(Qt.PenStyle.NoPen))
            rect.setZValue(-10)
            self.addItem(rect)
        for i in range(num_tracks + 1):
            line = QGraphicsLineItem(HEADER_WIDTH, i * TRACK_HEIGHT,
                                     10000, i * TRACK_HEIGHT)
            line.setPen(QPen(QColor(COLORS["border"]), 1))
            line.setZValue(-5)
            self.addItem(line)
        header_bg = QGraphicsRectItem(0, 0, HEADER_WIDTH,
                                      num_tracks * TRACK_HEIGHT)
        header_bg.setBrush(QBrush(QColor(COLORS["bg_alt"])))
        header_bg.setPen(QPen(QColor(COLORS["border"]), 1))
        header_bg.setZValue(0)
        self.addItem(header_bg)
        self.playhead = PlayheadItem(num_tracks)
        self.addItem(self.playhead)
        self.clips = []

    def add_clip(self, stem_name, audio_data, samplerate, track_index):
        c = ClipItem(stem_name, audio_data, samplerate, track_index)
        self.addItem(c)
        self.clips.append(c)
        return c

    def set_playhead(self, seconds):
        self.playhead.set_position(
            HEADER_WIDTH + 8 + seconds * PIXELS_PER_SECOND)

    def clear_clips(self):
        for c in self.clips:
            self.removeItem(c)
        self.clips.clear()


class TimelineView(QGraphicsView):
    zoom_changed = pyqtSignal(float)

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setStyleSheet(
            "QGraphicsView { background-color: " + COLORS["bg"] +
            "; border: 1px solid " + COLORS["border"] +
            "; border-radius: 8px; }")
        self._zoom = 1.0

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self._zoom *= factor
            self.scale(factor, 1.0)
            self.zoom_changed.emit(self._zoom)
            event.accept()
        else:
            super().wheelEvent(event)


class TrackHeader(QWidget):
    mute_toggled = pyqtSignal(str, bool)
    solo_toggled = pyqtSignal(str, bool)
    volume_changed = pyqtSignal(str, float)

    def __init__(self, stem_name, color_hex, parent=None):
        super().__init__(parent)
        self.stem_name = stem_name
        self.setFixedHeight(TRACK_HEIGHT - 16)
        self.setStyleSheet(
            "QWidget { background-color: " + COLORS["bg_alt"] +
            "; border-bottom: 1px solid " + COLORS["border"] + "; "
            "border-left: 4px solid " + color_hex + "; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(4)
        name = QLabel(stem_name.capitalize())
        name.setStyleSheet(
            "font-weight: 700; font-size: 12px; color: " + color_hex + ";")
        layout.addWidget(name)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.mute_btn = QPushButton("M")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setFixedSize(24, 22)
        self.mute_btn.setStyleSheet(
            "QPushButton { background-color: " + COLORS["bg_card"] +
            "; color: " + COLORS["text_dim"] +
            "; border: 1px solid " + COLORS["border"] +
            "; border-radius: 4px; font-weight: 600; font-size: 11px; }"
            "QPushButton:checked { background-color: #c94d4d; "
            "color: #ffffff; border-color: #e06666; }")
        self.mute_btn.toggled.connect(
            lambda checked: self.mute_toggled.emit(
                self.stem_name, checked))
        btn_row.addWidget(self.mute_btn)
        self.solo_btn = QPushButton("S")
        self.solo_btn.setCheckable(True)
        self.solo_btn.setFixedSize(24, 22)
        self.solo_btn.setStyleSheet(
            "QPushButton { background-color: " + COLORS["bg_card"] +
            "; color: " + COLORS["text_dim"] +
            "; border: 1px solid " + COLORS["border"] +
            "; border-radius: 4px; font-weight: 600; font-size: 11px; }"
            "QPushButton:checked { background-color: #f0b655; "
            "color: #1e1e22; border-color: #f5c977; }")
        self.solo_btn.toggled.connect(
            lambda checked: self.solo_toggled.emit(
                self.stem_name, checked))
        btn_row.addWidget(self.solo_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 200)
        self.vol_slider.setValue(100)
        self.vol_slider.valueChanged.connect(
            lambda v: self.volume_changed.emit(
                self.stem_name, v / 100.0))
        layout.addWidget(self.vol_slider)
FILE_END

# ---------------------------------------------------------------------------
# 5. ui/workers.py — append BackgroundExportWorker if missing
# ---------------------------------------------------------------------------
echo ""
echo ">> [5/7] ui/workers.py"
backup "ui/workers.py"
if grep -q "class BackgroundExportWorker" ui/workers.py; then
    echo "   BackgroundExportWorker already present."
else
    cat >> ui/workers.py <<'FILE_END'


class BackgroundExportWorker(QThread):
    """Runs export without blocking playback."""
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, tracks, dsp_chains, output_path,
                 fmt_key, bitrate):
        super().__init__()
        self.tracks = tracks
        self.dsp_chains = dsp_chains
        self.output_path = output_path
        self.fmt_key = fmt_key
        self.bitrate = bitrate

    def run(self):
        from core.export import (render_mix_to_wav,
                                 encode_with_ffmpeg)
        tmp = self.output_path + ".tmp.wav"
        try:
            def p1(p):
                self.progress.emit(int(p * 0.6))
            ok = render_mix_to_wav(
                self.tracks, self.dsp_chains, tmp,
                progress_cb=p1)
            if not ok:
                self.failed.emit("Render failed.")
                return
            def p2(p):
                self.progress.emit(60 + int(p * 0.4))
            ok = encode_with_ffmpeg(
                tmp, self.output_path, self.fmt_key,
                self.bitrate, progress_cb=p2)
            if not ok:
                self.failed.emit("Encode failed.")
                return
            self.finished_ok.emit(self.output_path)
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
FILE_END
    echo "   BackgroundExportWorker appended."
fi

# ---------------------------------------------------------------------------
# 6. ui/dialogs.py — speed hints (string replace)
# ---------------------------------------------------------------------------
echo ""
echo ">> [6/7] ui/dialogs.py"
backup "ui/dialogs.py"
python3 - <<'PYEOF'
path = "ui/dialogs.py"
with open(path, "r") as f:
    src = f.read()

if "MLX-accelerated on Apple Silicon" in src:
    print("   already updated.")
else:
    old_ht = ('"Fast 4-stem: single file, ~30% faster.")')
    new_ht = ('"Fast 4-stem: MLX-accelerated on Apple Silicon. "\n'
              '                "Leave \'Force CPU\' unchecked for speed.")')
    old_ft = ('"High Quality 4-stem: best separation quality.")')
    new_ft = ('"High Quality 4-stem: MLX-accelerated on Apple "\n'
              '                "Silicon. Leave \'Force CPU\' unchecked.")')
    if old_ht in src:
        src = src.replace(old_ht, new_ht)
    if old_ft in src:
        src = src.replace(old_ft, new_ft)
    with open(path, "w") as f:
        f.write(src)
    print("   speed hints updated.")
PYEOF

# ---------------------------------------------------------------------------
# 7. apps/daw.py — full new DAW
# ---------------------------------------------------------------------------
echo ""
echo ">> [7/7] apps/daw.py"
backup "apps/daw.py"
cat > apps/daw.py <<'FILE_END'
"""StemsMix DAW - real timeline engine with clip editing."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QDialog, QScrollArea,
                             QSlider, QDoubleSpinBox, QGroupBox)
from PyQt6.QtCore import Qt, QTimer

from ui.style import COLORS, QSS
from ui.dialogs import PrerequisiteDialog, ExportDialog
from ui.workers import (DrumSplitWorker, ChordAnalysisWorker)
from ui.timeline import (TimelineScene, TimelineView, TrackHeader,
                         HEADER_WIDTH, TRACK_HEIGHT, PIXELS_PER_SECOND)
from ui.channel_strip import ChannelStrip
from core.resources import run_prerequisite_checks
from core.dsp import DSPChain
from core.export import EXPORT_FORMATS, encode_with_ffmpeg
from core.audio_io import load_audio
from core.timeline_model import Clip, Track, Project
from core.engine import TimelineEngine


STEM_NAMES = ('vocals', 'drums', 'bass', 'other', 'guitar', 'piano')
DRUM_PARTS = ('kick', 'snare', 'toms', 'hh', 'ride', 'crash')


class DAWWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StemsMix DAW")
        self.setGeometry(100, 100, 1500, 950)
        self.project = Project()
        self.dsp_chains = {}
        self.track_headers = []
        self.selected_clip = None
        self.engine = None
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(40)
        self.playback_timer.timeout.connect(self.update_playhead)
        self.export_worker = None
        self.drum_worker = None
        self.chord_worker = None
        self.bg_export = None
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
        sub = QLabel("Timeline, Clip Editing, Effects")
        sub.setObjectName("header_subtitle")
        hl.addWidget(sub)
        hl.addStretch()
        main.addWidget(header)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self.btn_load = QPushButton("Load Stems Folder...")
        self.btn_load.clicked.connect(self.load_stems)
        toolbar.addWidget(self.btn_load)
        self.btn_split_drums = QPushButton("Split Drums")
        self.btn_split_drums.setEnabled(False)
        self.btn_split_drums.clicked.connect(self.split_drums)
        toolbar.addWidget(self.btn_split_drums)
        self.btn_chords = QPushButton("Detect Chords")
        self.btn_chords.setEnabled(False)
        self.btn_chords.clicked.connect(self.detect_chords)
        toolbar.addWidget(self.btn_chords)
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

        timeline_row = QHBoxLayout()
        timeline_row.setSpacing(0)
        self.headers_scroll = QScrollArea()
        self.headers_scroll.setFixedWidth(HEADER_WIDTH)
        self.headers_scroll.setWidgetResizable(True)
        self.headers_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.headers_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.headers_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }")
        self.headers_inner = QWidget()
        self.headers_inner.setStyleSheet(
            "QWidget { background-color: " + COLORS["bg"] + "; }")
        self.headers_layout = QVBoxLayout(self.headers_inner)
        self.headers_layout.setContentsMargins(0, 0, 0, 0)
        self.headers_layout.setSpacing(8)
        self.headers_scroll.setWidget(self.headers_inner)
        timeline_row.addWidget(self.headers_scroll)
        self.scene = TimelineScene(num_tracks=6)
        self.scene.selectionChanged.connect(self.on_selection_changed)
        self.view = TimelineView(self.scene)
        self.view.setMinimumHeight(400)
        timeline_row.addWidget(self.view, 1)
        main.addLayout(timeline_row, 3)

        self.props_group = QGroupBox("Selected Clip")
        props_layout = QHBoxLayout(self.props_group)
        props_layout.setContentsMargins(12, 8, 12, 8)
        props_layout.setSpacing(16)
        self.lbl_clip_name = QLabel("(no clip selected)")
        self.lbl_clip_name.setStyleSheet(
            "font-weight: 600; color: " + COLORS["text"] + ";")
        props_layout.addWidget(self.lbl_clip_name)
        props_layout.addWidget(QLabel("Start:"))
        self.spin_start = QDoubleSpinBox()
        self.spin_start.setRange(0.0, 3600.0)
        self.spin_start.setSuffix(" s")
        self.spin_start.setDecimals(2)
        self.spin_start.valueChanged.connect(self.on_clip_start_changed)
        props_layout.addWidget(self.spin_start)
        props_layout.addWidget(QLabel("Stretch:"))
        self.spin_stretch = QDoubleSpinBox()
        self.spin_stretch.setRange(0.5, 2.0)
        self.spin_stretch.setSingleStep(0.05)
        self.spin_stretch.setDecimals(2)
        self.spin_stretch.setValue(1.0)
        self.spin_stretch.valueChanged.connect(self.on_clip_stretch_changed)
        props_layout.addWidget(self.spin_stretch)
        props_layout.addWidget(QLabel("Pitch:"))
        self.spin_pitch = QDoubleSpinBox()
        self.spin_pitch.setRange(0.5, 2.0)
        self.spin_pitch.setSingleStep(0.01)
        self.spin_pitch.setDecimals(3)
        self.spin_pitch.setValue(1.0)
        self.spin_pitch.valueChanged.connect(self.on_clip_pitch_changed)
        props_layout.addWidget(self.spin_pitch)
        props_layout.addWidget(QLabel("Gain:"))
        self.slider_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_gain.setRange(0, 200)
        self.slider_gain.setValue(100)
        self.slider_gain.valueChanged.connect(self.on_clip_gain_changed)
        props_layout.addWidget(self.slider_gain, 1)
        main.addWidget(self.props_group)

        self.strips_container = QWidget()
        self.strips_layout = QHBoxLayout(self.strips_container)
        self.strips_layout.setContentsMargins(0, 0, 0, 0)
        self.strips_layout.setSpacing(8)
        main.addWidget(self.strips_container, 1)

        self.status = QLabel("Load a stems folder to begin.")
        self.status.setObjectName("status_bar")
        self.status.setWordWrap(True)
        main.addWidget(self.status)

        self.view.verticalScrollBar().valueChanged.connect(
            self.headers_scroll.verticalScrollBar().setValue)

    def load_stems(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Stems Folder")
        if not folder:
            return
        self.project = Project()
        loaded = 0
        for name in STEM_NAMES:
            p = os.path.join(folder, name + ".wav")
            if not os.path.exists(p):
                continue
            try:
                data, sr = load_audio(p)
            except Exception as e:
                print("Failed {}: {}".format(name, e))
                continue
            self.project.samplerate = sr
            track = Track(name, COLORS["track_colors"].get(
                name, COLORS["accent"]))
            clip = Clip(name, p, data, sr, start_time=0.0)
            track.add_clip(clip)
            self.project.add_track(track)
            self.dsp_chains[name] = DSPChain(sr)
            loaded += 1
        if loaded == 0:
            self.status.setText("No valid stems found.")
            return
        self._rebuild_ui()
        self.status.setText(
            "Loaded {} stems from {}".format(loaded, folder))

    def _rebuild_ui(self):
        self.scene.clear_clips()
        self.scene.num_tracks = len(self.project.tracks)
        for i, track in enumerate(self.project.tracks):
            for clip in track.clips:
                scene_clip = self.scene.add_clip(
                    clip.stem_name, clip.data, clip.samplerate, i)
                scene_clip.clip_ref = clip

        while self.headers_layout.count():
            item = self.headers_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.track_headers = []
        for track in self.project.tracks:
            color = COLORS["track_colors"].get(
                track.name, COLORS["accent"])
            header = TrackHeader(track.name, color)
            header.mute_toggled.connect(self.set_track_mute)
            header.solo_toggled.connect(self.set_track_solo)
            header.volume_changed.connect(self.set_track_volume)
            self.headers_layout.addWidget(header)
            self.track_headers.append(header)
        self.headers_layout.addStretch()

        while self.strips_layout.count():
            item = self.strips_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for track in self.project.tracks:
            self.strips_layout.addWidget(
                ChannelStrip(track.name, self.dsp_chains[track.name]))

        if self.engine:
            self.engine.stop()
        self.engine = TimelineEngine(
            self.project, lambda name: self.dsp_chains[name])

        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_split_drums.setEnabled(
            any(t.name == "drums" for t in self.project.tracks))
        self.btn_chords.setEnabled(
            any(t.name in ("other", "guitar", "piano", "vocals")
                for t in self.project.tracks))

    def set_track_mute(self, name, muted):
        for t in self.project.tracks:
            if t.name == name:
                t.muted = muted

    def set_track_solo(self, name, soloed):
        for t in self.project.tracks:
            if t.name == name:
                t.soloed = soloed

    def set_track_volume(self, name, volume):
        for t in self.project.tracks:
            if t.name == name:
                t.volume = volume

    def on_selection_changed(self):
        items = self.scene.selectedItems()
        clip_items = [i for i in items if hasattr(i, "clip_ref")]
        if not clip_items or clip_items[0].clip_ref is None:
            self.selected_clip = None
            self.lbl_clip_name.setText("(no clip selected)")
            return
        clip = clip_items[0].clip_ref
        self.selected_clip = (clip, clip_items[0])
        self.lbl_clip_name.setText(clip.stem_name.capitalize())
        for w, v in [(self.spin_start, clip.start_time),
                     (self.spin_stretch, clip.stretch),
                     (self.spin_pitch, clip.pitch)]:
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self.slider_gain.blockSignals(True)
        self.slider_gain.setValue(int(clip.gain * 100))
        self.slider_gain.blockSignals(False)

    def on_clip_start_changed(self, value):
        if self.selected_clip:
            clip, item = self.selected_clip
            clip.start_time = value
            item.setX(HEADER_WIDTH + 8 + value * PIXELS_PER_SECOND)
            self.project._recompute_duration()

    def on_clip_stretch_changed(self, value):
        if self.selected_clip:
            clip, _ = self.selected_clip
            clip.stretch = value
            self.project._recompute_duration()

    def on_clip_pitch_changed(self, value):
        if self.selected_clip:
            clip, _ = self.selected_clip
            clip.pitch = value

    def on_clip_gain_changed(self, value):
        if self.selected_clip:
            clip, _ = self.selected_clip
            clip.gain = value / 100.0

    def toggle_playback(self):
        if not self.engine:
            return
        if self.engine.is_playing:
            self.engine.stop()
            self.playback_timer.stop()
            self.btn_play.setText("Play")
        else:
            self.engine.start(from_seconds=self.engine.playhead_seconds)
            self.playback_timer.start()
            self.btn_play.setText("Pause")

    def stop_playback(self):
        if self.engine:
            self.engine.stop()
            self.engine.seek(0)
        self.playback_timer.stop()
        self.scene.set_playhead(0)
        self.btn_play.setText("Play")
        self.status.setText("Stopped.")

    def update_playhead(self):
        if self.engine and self.engine.is_playing:
            self.scene.set_playhead(self.engine.playhead_seconds)

    def export_mix(self):
        if not self.project.tracks:
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
        tmp_wav = path + ".tmp.wav"
        self.status.setText("Rendering...")
        QApplication.processEvents()
        ok = self.engine.render_offline(tmp_wav)
        if not ok:
            self.status.setText("Render failed.")
            return
        ok = encode_with_ffmpeg(tmp_wav, path, fmt, br)
        try:
            os.remove(tmp_wav)
        except OSError:
            pass
        if ok:
            self.status.setText("Exported: " + path)
        else:
            self.status.setText("Encode failed.")

    def split_drums(self):
        drums_track = next(
            (t for t in self.project.tracks if t.name == "drums"),
            None)
        if not drums_track or not drums_track.clips:
            self.status.setText("No drums clip loaded.")
            return
        drums_path = drums_track.clips[0].filepath
        out_dir = os.path.join(
            os.path.dirname(drums_path), "drum_parts")
        os.makedirs(out_dir, exist_ok=True)
        self.btn_split_drums.setEnabled(False)
        self.status.setText("Splitting drums...")
        self.drum_worker = DrumSplitWorker(
            drums_path, out_dir, device="cpu")
        self.drum_worker.progress.connect(
            lambda p, m: self.status.setText(m))
        self.drum_worker.finished_ok.connect(
            lambda paths: self.on_drums_split(paths, out_dir))
        self.drum_worker.failed.connect(
            lambda e: self.status.setText("Drum split failed: " + e))
        self.drum_worker.finished.connect(
            lambda: self.btn_split_drums.setEnabled(True))
        self.drum_worker.start()

    def on_drums_split(self, paths, out_dir):
        self.project.tracks = [
            t for t in self.project.tracks if t.name != "drums"]
        for name in DRUM_PARTS:
            p = os.path.join(out_dir, name + ".wav")
            if not os.path.exists(p):
                continue
            try:
                data, sr = load_audio(p)
            except Exception as e:
                print("Failed {}: {}".format(name, e))
                continue
            track = Track(name, COLORS["track_colors"].get(
                name, COLORS["accent"]))
            track.add_clip(Clip(name, p, data, sr, start_time=0.0))
            self.project.add_track(track)
            if name not in self.dsp_chains:
                self.dsp_chains[name] = DSPChain(sr)
        self._rebuild_ui()
        self.status.setText(
            "Drums split into {} parts.".format(len(DRUM_PARTS)))

    def detect_chords(self):
        harmonic = None
        for pref in ("other", "guitar", "piano", "vocals"):
            t = next(
                (x for x in self.project.tracks if x.name == pref),
                None)
            if t and t.clips:
                harmonic = t.clips[0].filepath
                break
        if not harmonic:
            self.status.setText("No harmonic stem available.")
            return
        self.btn_chords.setEnabled(False)
        self.status.setText("Analyzing chords...")
        self.chord_worker = ChordAnalysisWorker(harmonic)
        self.chord_worker.progress.connect(
            lambda p, m: self.status.setText(m))
        self.chord_worker.finished_ok.connect(
            self.on_chords_detected)
        self.chord_worker.failed.connect(
            lambda e: self.status.setText("Chord detection failed: " + e))
        self.chord_worker.finished.connect(
            lambda: self.btn_chords.setEnabled(True))
        self.chord_worker.start()

    def on_chords_detected(self, chords):
        if not chords:
            self.status.setText("No chords detected.")
            return
        sample = ", ".join(
            "{:.1f}s:{}".format(t, c) for t, c in chords[:10])
        suffix = " ..." if len(chords) > 10 else ""
        self.status.setText(
            "Chords ({}): {}{}".format(len(chords), sample, suffix))

    def closeEvent(self, event):
        if self.engine:
            self.engine.stop()
        self.playback_timer.stop()
        for w in (self.export_worker, self.drum_worker,
                  self.chord_worker, self.bg_export):
            if w is not None and w.isRunning():
                w.terminate()
                w.wait(2000)
        event.accept()


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

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
echo ""
echo ">> Verifying syntax..."
FAIL=0
for f in core/timeline_model.py core/engine.py core/separation.py \
         ui/timeline.py ui/workers.py ui/dialogs.py apps/daw.py; do
    if python3 -c "import ast; ast.parse(open('$f').read())" 2>/dev/null; then
        echo "   $f: OK"
    else
        echo "   $f: SYNTAX ERROR"
        python3 -c "import ast; ast.parse(open('$f').read())" || true
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo "One or more files failed. Backups at *.bak.$STAMP"
    exit 1
fi

# ---------------------------------------------------------------------------
# Optional dependency check
# ---------------------------------------------------------------------------
echo ""
echo ">> Checking optional dependencies..."
python3 -c "
mods = [
    ('demucs_onnx', 'demucs-onnx'),
    ('demucs_mlx', 'demucs-mlx (MLX acceleration)'),
    ('mdxnet_infer', 'mdxnet-infer (drum split)'),
    ('denote', 'denote (chord detection)'),
    ('celeritas', 'celeritas (chord names)'),
]
for mod, label in mods:
    try:
        __import__(mod)
        print('   ' + label + ': OK')
    except ImportError:
        print('   ' + label + ': NOT INSTALLED')
" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=================================================="
echo " All modifications applied."
echo ""
echo " Files changed:"
echo "   core/timeline_model.py    (new)"
echo "   core/engine.py            (new)"
echo "   core/separation.py        (MLX-first)"
echo "   ui/timeline.py            (waveforms, headers)"
echo "   ui/workers.py             (background export)"
echo "   ui/dialogs.py             (speed hints)"
echo "   apps/daw.py               (full rewrite)"
echo ""
echo " Backups: *.bak.$STAMP"
echo " Roll back: cp <file>.bak.$STAMP <file>"
echo ""
echo " Run the app:  python apps/daw.py"
echo "=================================================="
