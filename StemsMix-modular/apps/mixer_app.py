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
                             QCheckBox, QProgressBar, QDialog)
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
