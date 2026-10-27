"""StemsMix DAW - timeline, effects, drum splitting, chord detection."""
import sys
import os
import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QSplitter, QDialog)
from PyQt6.QtCore import Qt, QTimer

from ui.style import COLORS, QSS
from ui.dialogs import PrerequisiteDialog, ExportDialog
from ui.workers import (ExportWorker, DrumSplitWorker,
                        ChordAnalysisWorker)
from ui.timeline import TimelineScene, TimelineView
from ui.channel_strip import ChannelStrip
from core.resources import run_prerequisite_checks
from core.dsp import DSPChain
from core.export import EXPORT_FORMATS
from core.audio_io import load_audio


STEM_NAMES = ('vocals', 'drums', 'bass', 'other', 'guitar', 'piano')
DRUM_PARTS = ('kick', 'snare', 'toms', 'hh', 'ride', 'crash')


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
        self.export_worker = None
        self.drum_worker = None
        self.chord_worker = None
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
        sub = QLabel("Timeline, Effects, Drum Split, Chords")
        sub.setObjectName("header_subtitle")
        hl.addWidget(sub)
        hl.addStretch()
        main.addWidget(header)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.btn_load = QPushButton("Load Stems Folder...")
        self.btn_load.clicked.connect(self.load_stems)
        toolbar.addWidget(self.btn_load)

        self.btn_split_drums = QPushButton("Split Drums Further")
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

        self.status = QLabel("Load a stems folder to begin.")
        self.status.setObjectName("status_bar")
        self.status.setWordWrap(True)
        main.addWidget(self.status)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

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
            "Loaded {} stems from {}".format(len(loaded), folder))

    def _install_tracks(self, tracks):
        self.tracks = tracks
        self.dsp_chains = [DSPChain(t.samplerate) for t in tracks]
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

        # Enable drum split only if a drums track is present
        has_drums = any(t.name == "drums" for t in tracks)
        self.btn_split_drums.setEnabled(has_drums)

        # Enable chord detection if any harmonic stem is present
        has_harmonic = any(
            t.name in ("other", "guitar", "piano", "vocals")
            for t in tracks)
        self.btn_chords.setEnabled(has_harmonic)

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

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

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

        self.export_worker = ExportWorker(
            self.tracks, self.dsp_chains, path, fmt, br)
        self.export_worker.progress.connect(
            lambda p: self.status.setText(
                "Exporting... {}%".format(p)))
        self.export_worker.finished_ok.connect(
            lambda p: self.status.setText("Exported: " + p))
        self.export_worker.failed.connect(
            lambda e: self.status.setText("Export failed: " + e))
        self.export_worker.start()

    # ------------------------------------------------------------------
    # Drum splitting
    # ------------------------------------------------------------------

    def split_drums(self):
        drums_path = None
        for t in self.tracks:
            if t.name == "drums":
                drums_path = t.filepath
                break
        if not drums_path:
            self.status.setText("No drums stem loaded.")
            return

        out_dir = os.path.join(
            os.path.dirname(drums_path), "drum_parts")
        os.makedirs(out_dir, exist_ok=True)

        self.btn_split_drums.setEnabled(False)
        self.status.setText("Splitting drums...")

        # device="cpu" is required on M1 (mdxnet-infer has no MPS support).
        # Change to "cuda" on a machine with an NVIDIA GPU.
        self.drum_worker = DrumSplitWorker(
            drums_path, out_dir, device="cpu")
        self.drum_worker.progress.connect(
            lambda p, m: self.status.setText(m))
        self.drum_worker.finished_ok.connect(
            lambda paths: self.on_drums_split(paths, out_dir))
        self.drum_worker.failed.connect(
            lambda e: self.status.setText(
                "Drum split failed: " + e))
        self.drum_worker.finished.connect(
            lambda: self.btn_split_drums.setEnabled(True))
        self.drum_worker.start()

    def on_drums_split(self, paths, out_dir):
        new_tracks = []
        for name in DRUM_PARTS:
            p = os.path.join(out_dir, name + ".wav")
            if os.path.exists(p):
                try:
                    new_tracks.append(StemTrack(p, name))
                except Exception as e:
                    print("Failed {}: {}".format(name, e))

        if not new_tracks:
            self.status.setText("Drum split produced no files.")
            return

        # Replace the mixed drums track with the individual parts
        base_tracks = [t for t in self.tracks if t.name != "drums"]
        all_tracks = base_tracks + new_tracks
        self._install_tracks(all_tracks)
        self.status.setText(
            "Added {} drum parts. Total tracks: {}".format(
                len(new_tracks), len(all_tracks)))

    # ------------------------------------------------------------------
    # Chord detection
    # ------------------------------------------------------------------

    def detect_chords(self):
        harmonic_path = None
        for preferred in ("other", "guitar", "piano", "vocals"):
            for t in self.tracks:
                if t.name == preferred:
                    harmonic_path = t.filepath
                    break
            if harmonic_path:
                break

        if not harmonic_path:
            self.status.setText(
                "No harmonic stem available for chord detection.")
            return

        self.btn_chords.setEnabled(False)
        self.status.setText("Analyzing chords...")

        self.chord_worker = ChordAnalysisWorker(harmonic_path)
        self.chord_worker.progress.connect(
            lambda p, m: self.status.setText(m))
        self.chord_worker.finished_ok.connect(
            self.on_chords_detected)
        self.chord_worker.failed.connect(
            lambda e: self.status.setText(
                "Chord detection failed: " + e))
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

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self.mixer:
            self.mixer.stop()
        self.playback_timer.stop()
        for w in (self.export_worker, self.drum_worker,
                  self.chord_worker):
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
