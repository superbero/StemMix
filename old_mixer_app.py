# mixer_app.py
import sys
import os
import traceback
import numpy as np
import sounddevice as sd
import soundfile as sf
import stemextract
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QLabel,
                             QFileDialog, QGroupBox, QCheckBox, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal


# ---------------------------------------------------------------------------
# Background separation worker
# ---------------------------------------------------------------------------

class SeparationWorker(QThread):
    """Runs stemextract.extract() off the GUI thread."""

    finished_ok = pyqtSignal(dict)   # emitted with the dict of stem paths
    failed = pyqtSignal(str)         # emitted with an error message

    def __init__(self, input_file, output_dir, segment=5):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.segment = segment

    def run(self):
        try:
            # device="auto" -> CUDA if available, else CPU
            result = stemextract.extract(
                self.input_file,
                self.output_dir,
                segment=self.segment,
            )
            self.finished_ok.emit(result)
        except Exception as e:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr)
            self.failed.emit(str(e))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class StemTrack:
    """Holds audio data and state for a single stem."""

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

    def start(self):
        if self.stream is not None:
            self.stop()

        self.stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=2,
            callback=self._callback,
            blocksize=1024,
        )
        self.stream.start()
        self.is_playing = True

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False

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

        mix = np.clip(mix, -1.0, 1.0)
        outdata[:] = mix


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_mix(tracks, output_path, samplerate=44100):
    active_tracks = [t for t in tracks if not t.muted]
    if not active_tracks:
        return False

    max_len = max(len(t.data) for t in active_tracks)
    mix = np.zeros((max_len, 2), dtype='float32')

    for track in active_tracks:
        chunk = track.data * track.volume
        mix[:len(chunk)] += chunk

    mix = np.clip(mix, -1.0, 1.0)

    if mix.ndim == 1:
        mix = np.column_stack((mix, mix))

    sf.write(output_path, mix, samplerate, subtype='PCM_24')
    return True


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

STEM_NAMES = ('vocals', 'drums', 'bass', 'other')


class MixerWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python Stem Mixer - Moises Alternative")
        self.setGeometry(100, 100, 950, 650)

        self.tracks = []
        self.mixer = None
        self.worker = None

        self.init_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- Top controls ---
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

        # --- Progress bar (hidden unless separation is running) ---
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate spinner
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)

        # --- Track controls ---
        self.tracks_group = QGroupBox("Tracks")
        self.tracks_layout = QVBoxLayout()
        self.tracks_group.setLayout(self.tracks_layout)
        main_layout.addWidget(self.tracks_group)

        # --- Status bar ---
        self.status_label = QLabel(
            "Open an audio file to separate, or load a folder of existing stems."
        )
        main_layout.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Separation
    # ------------------------------------------------------------------

    def open_audio_file(self):
        if self.worker is not None and self.worker.isRunning():
            self.status_label.setText("Separation already running, please wait.")
            return

        input_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            os.path.expanduser("~"),
            "Audio Files (*.mp3 *.wav *.m4a *.flac *.ogg);;All Files (*)",
        )
        if not input_file:
            return

        default_out = os.path.splitext(input_file)[0] + "_stems"
        output_dir = QFileDialog.getExistingDirectory(
            self,
            f"Choose output folder (default: {default_out})",
            os.path.dirname(input_file),
        )
        if not output_dir:
            # User cancelled -> fall back to default next to input
            output_dir = default_out
        os.makedirs(output_dir, exist_ok=True)

        self._set_separation_running(True)
        self.status_label.setText(
            f"Separating '{os.path.basename(input_file)}' — this may take a while…"
        )

        self.worker = SeparationWorker(input_file, output_dir)
        self.worker.finished_ok.connect(lambda paths: self.on_separation_done(paths, output_dir))
        self.worker.failed.connect(self.on_separation_failed)
        self.worker.finished.connect(lambda: self._set_separation_running(False))
        self.worker.start()

    def _set_separation_running(self, running):
        self.progress.setVisible(running)
        self.btn_open.setEnabled(not running)
        self.btn_load.setEnabled(not running)

    def on_separation_done(self, result_dict, output_dir):
        # stemextract returns a dict; the actual files are on disk as <name>.wav.
        # We just load from the folder, same as the "Load Stems Folder" button.
        self._load_stems_from_folder(output_dir)
        self.status_label.setText(
            f"Separation complete. Loaded {len(self.tracks)} stems from {output_dir}"
        )

    def on_separation_failed(self, message):
        self.status_label.setText(f"Separation failed: {message}")
        print(f"Separation failed: {message}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Loading stems
    # ------------------------------------------------------------------

    def load_stems_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Stems Folder")
        if not folder:
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
            else:
                print(f"File not found: {path}", file=sys.stderr)

        if not loaded:
            self.status_label.setText(
                "No valid stem files found (expected vocals.wav, drums.wav, bass.wav, other.wav)."
            )
            return

        self.tracks = loaded
        self.rebuild_track_controls()

        if self.mixer:
            self.mixer.stop()
        self.mixer = AudioMixer(self.tracks)

        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_play.setText("Play")
        self.status_label.setText(f"Loaded {len(self.tracks)} stems from {folder}")

    # ------------------------------------------------------------------
    # Track controls UI
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
            self.btn_play.setText("Play")
        else:
            for track in self.tracks:
                track.reset()
            self.mixer.start()
            self.btn_play.setText("Pause")

    def stop_playback(self):
        if self.mixer:
            self.mixer.stop()
        for track in self.tracks:
            track.reset()
        self.btn_play.setText("Play")
        self.status_label.setText("Stopped and reset to beginning.")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_mix_dialog(self):
        if not self.tracks:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Mix", "mix.wav", "WAV Files (*.wav)"
        )
        if not path:
            return

        success = export_mix(self.tracks, path, samplerate=self.tracks[0].samplerate)
        if success:
            self.status_label.setText(f"Exported mix to: {path}")
        else:
            self.status_label.setText("Export failed — no active tracks.")

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self.mixer:
            self.mixer.stop()
        if self.worker is not None and self.worker.isRunning():
            # We can't cleanly cancel stemextract, but at least let the app close.
            self.worker.terminate()
            self.worker.wait(2000)
        event.accept()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MixerWindow()
    window.show()
    sys.exit(app.exec())
