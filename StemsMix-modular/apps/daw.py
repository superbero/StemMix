"""StemsMix DAW - real timeline engine with clip editing."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QDialog, QScrollArea,
                             QSlider, QDoubleSpinBox, QGroupBox,
                             QProgressBar)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal

from ui.style import COLORS, QSS
from ui.dialogs import (PrerequisiteDialog, ExportDialog,
                        ModelSelectionDialog)
from ui.workers import (DrumSplitWorker, ChordAnalysisWorker,
                        SeparationWorker)
from ui.timeline import (TimelineScene, TimelineView, TrackHeader,
                         HEADER_WIDTH, TRACK_HEIGHT, PIXELS_PER_SECOND)
from ui.channel_strip import ChannelStrip
from core.resources import (run_prerequisite_checks,
                            check_system_resources)
from core.dsp import DSPChain
from core.export import EXPORT_FORMATS, encode_with_ffmpeg
from core.audio_io import load_audio
from core.timeline_model import Clip, Track, Project
from core.engine import TimelineEngine


STEM_NAMES = ('vocals', 'drums', 'bass', 'other', 'guitar', 'piano')
DRUM_PARTS = ('kick', 'snare', 'toms', 'hh', 'ride', 'crash')


class RenderExportWorker(QThread):
    """Renders the timeline and encodes it off the UI thread."""
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, engine, path, fmt, bitrate, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.path = path
        self.fmt = fmt
        self.bitrate = bitrate

    def run(self):
        tmp_wav = self.path + ".tmp.wav"
        try:
            self.progress.emit(5, "Rendering mix...")
            if not self.engine.render_offline(tmp_wav):
                self.failed.emit("Nothing to render.")
                return
            self.progress.emit(
                50, "Encoding...")
            ok = encode_with_ffmpeg(
                tmp_wav, self.path, self.fmt, self.bitrate,
                progress_cb=lambda p: self.progress.emit(
                    50 + p // 2, "Encoding..."))
            if ok:
                self.finished_ok.emit(self.path)
            else:
                self.failed.emit("Encode failed.")
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            try:
                os.remove(tmp_wav)
            except OSError:
                pass


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
        self.sep_worker = None
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
        self.btn_open = QPushButton("Open Audio && Separate...")
        self.btn_open.setObjectName("primary")
        self.btn_open.clicked.connect(self.open_audio_file)
        toolbar.addWidget(self.btn_open)
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
        # Must match the timeline lanes: header height (TRACK_HEIGHT - 16)
        # + spacing 16 = TRACK_HEIGHT per track, with the same 8 px top
        # offset the clips use. With spacing 8 the headers drifted 8 px
        # further from their lanes on every track.
        self.headers_layout.setContentsMargins(0, 8, 0, 0)
        self.headers_layout.setSpacing(16)
        self.headers_scroll.setWidget(self.headers_inner)
        timeline_row.addWidget(self.headers_scroll)
        self.scene = TimelineScene(num_tracks=6)
        self.scene.selectionChanged.connect(self.on_selection_changed)
        self.scene.clip_moved.connect(self.on_clip_moved)
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

        self.progress_label = QLabel("")
        self.progress_label.setObjectName("progress_label")
        self.progress_label.setVisible(False)
        main.addWidget(self.progress_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        main.addWidget(self.progress)

        self.status = QLabel(
            "Open an audio file to separate, or load a stems folder.")
        self.status.setObjectName("status_bar")
        self.status.setWordWrap(True)
        main.addWidget(self.status)

        self.view.verticalScrollBar().valueChanged.connect(
            self.headers_scroll.verticalScrollBar().setValue)

    def load_stems(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Stems Folder")
        if folder:
            self._load_folder(folder)

    def _set_busy(self, busy):
        self.progress.setVisible(busy)
        self.progress_label.setVisible(busy)
        self.btn_open.setEnabled(not busy)
        self.btn_load.setEnabled(not busy)

    def open_audio_file(self):
        if self.sep_worker is not None and self.sep_worker.isRunning():
            self.status.setText("Separation already running.")
            return
        resources = check_system_resources()
        input_file, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", os.path.expanduser("~"),
            "Audio Files (*.mp3 *.wav *.m4a *.mp4 *.m4b *.aac "
            "*.flac *.ogg *.opus *.wma);;All Files (*)")
        if not input_file:
            return
        dlg = ModelSelectionDialog(resources, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        model = dlg.selected_model()
        providers = ["cpu"] if dlg.force_cpu() else "auto"
        chunked = dlg.use_chunked()

        out_dir = os.path.splitext(input_file)[0] + "_stems"
        if self.engine:
            self.engine.stop()
            self.playback_timer.stop()
            self.btn_play.setText("Play")
        self._set_busy(True)
        self.progress.setValue(0)
        self.progress_label.setText("Starting separation...")
        self.status.setText("Separating with " + model + "...")

        self.sep_worker = SeparationWorker(
            input_file, out_dir, model,
            providers=providers, chunked=chunked)
        self.sep_worker.progress.connect(self.on_sep_progress)
        self.sep_worker.finished_ok.connect(
            lambda r: self._load_folder(out_dir))
        self.sep_worker.failed.connect(
            lambda m: self.status.setText("Separation failed: " + m))
        self.sep_worker.finished.connect(
            lambda: self._set_busy(False))
        self.sep_worker.start()

    def on_sep_progress(self, pct, msg):
        self.progress.setValue(pct)
        self.progress_label.setText(msg)

    def _load_folder(self, folder):
        if self.engine:
            self.engine.stop()
            self.playback_timer.stop()
            self.btn_play.setText("Play")
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
        self.scene.set_extent(len(self.project.tracks),
                              self.project.total_duration)
        self.headers_inner.setMinimumHeight(
            len(self.project.tracks) * TRACK_HEIGHT + 20)
        for i, track in enumerate(self.project.tracks):
            for clip in track.clips:
                scene_clip = self.scene.add_clip(
                    clip.stem_name, clip.data, clip.samplerate, i)
                scene_clip.clip_ref = clip
                scene_clip.set_stretch(clip.stretch)
                scene_clip.setX(HEADER_WIDTH + 8
                                + clip.start_time * PIXELS_PER_SECOND)

        while self.headers_layout.count():
            item = self.headers_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.track_headers = []
        for track in self.project.tracks:
            color = COLORS["track_colors"].get(
                track.name, COLORS["accent"])
            header = TrackHeader(track.name, color)
            # Restore state before connecting, so rebuilding the UI
            # (e.g. after a drum split) doesn't show M/S/volume reset
            # while the tracks themselves are still muted/soloed.
            header.mute_btn.setChecked(track.muted)
            header.solo_btn.setChecked(track.soloed)
            header.vol_slider.setValue(int(round(track.volume * 100)))
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
        self._refresh_tool_buttons()

    def _refresh_tool_buttons(self):
        names = {t.name for t in self.project.tracks}
        self.btn_split_drums.setEnabled("drums" in names)
        self.btn_chords.setEnabled(
            bool(names & {"other", "guitar", "piano", "vocals"}))

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
            self.scene.ensure_duration(self.project.total_duration)

    def on_clip_moved(self):
        """A clip was dragged on the timeline: keep the project length
        (which ends playback) and the Start box in sync."""
        self.project._recompute_duration()
        self.scene.ensure_duration(self.project.total_duration)
        if self.selected_clip:
            clip, _ = self.selected_clip
            self.spin_start.blockSignals(True)
            self.spin_start.setValue(clip.start_time)
            self.spin_start.blockSignals(False)

    def on_clip_stretch_changed(self, value):
        if self.selected_clip:
            clip, item = self.selected_clip
            clip.stretch = value
            item.set_stretch(value)
            self.project._recompute_duration()
            self.scene.ensure_duration(self.project.total_duration)

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
        if self.export_worker is not None and self.export_worker.isRunning():
            self.status.setText("Export already running.")
            return
        # Offline render reuses the per-track DSP chains, so playback
        # must not run at the same time.
        if self.engine.is_playing:
            self.toggle_playback()
        self.btn_export.setEnabled(False)
        self.btn_play.setEnabled(False)
        self._set_busy(True)
        self.progress.setValue(0)
        self.export_worker = RenderExportWorker(
            self.engine, path, fmt, br)
        self.export_worker.progress.connect(self.on_sep_progress)
        self.export_worker.finished_ok.connect(
            lambda p: self.status.setText("Exported: " + p))
        self.export_worker.failed.connect(
            lambda m: self.status.setText("Export failed: " + m))
        self.export_worker.finished.connect(self._export_finished)
        self.export_worker.start()

    def _export_finished(self):
        self._set_busy(False)
        self.btn_export.setEnabled(True)
        self.btn_play.setEnabled(True)

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
        self.drum_worker.finished.connect(self._refresh_tool_buttons)
        self.drum_worker.start()

    def on_drums_split(self, paths, out_dir):
        drums = next((t for t in self.project.tracks
                      if t.name == "drums"), None)
        if drums is None:
            return
        src = drums.clips[0] if drums.clips else None

        # Use the files the worker actually wrote, not a hard-coded list.
        ordered = ([n for n in DRUM_PARTS if n in paths]
                   + [n for n in paths if n not in DRUM_PARTS])
        new_tracks = []
        for name in ordered:
            try:
                data, sr = load_audio(paths[name])
            except Exception as e:
                print("Failed {}: {}".format(name, e))
                continue
            track = Track(name, COLORS["track_colors"].get(
                name, COLORS["accent"]))
            clip = Clip(name, paths[name], data, sr,
                        start_time=src.start_time if src else 0.0)
            if src is not None:  # keep the parts aligned with the drums
                clip.stretch, clip.pitch, clip.gain = (
                    src.stretch, src.pitch, src.gain)
            track.add_clip(clip)
            track.muted, track.soloed, track.volume = (
                drums.muted, drums.soloed, drums.volume)
            new_tracks.append(track)
            self.project.samplerate = sr
            if name not in self.dsp_chains:
                self.dsp_chains[name] = DSPChain(sr)

        # Only replace the drums track once the parts really loaded.
        if not new_tracks:
            self.status.setText(
                "Drum split produced no usable parts; drums kept.")
            return
        idx = self.project.tracks.index(drums)
        self.project.tracks[idx:idx + 1] = new_tracks
        self.project._recompute_duration()
        self._rebuild_ui()
        self.status.setText(
            "Drums split into {} parts: {}".format(
                len(new_tracks), ", ".join(t.name for t in new_tracks)))

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
                  self.chord_worker, self.sep_worker):
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
