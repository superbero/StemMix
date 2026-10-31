"""StemsMix DAW: transport, arrange view, clip inspector, mixer."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from ui.project_actions import (install_shortcuts as install_project_shortcuts,
                                open_project_dialog, save_project_dialog)

from core.history import History


from core.stem_export import export_stems

from PyQt6.QtCore import Qt, QSize, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QDialog, QDoubleSpinBox, QFileDialog,
                             QFrame, QGridLayout, QHBoxLayout, QLabel,
                             QMainWindow, QProgressBar, QPushButton,
                             QSlider, QSplitter, QVBoxLayout, QWidget)

from ui.style import COLORS, QSS
from ui.dialogs import (PrerequisiteDialog, ExportDialog,
                        ModelSelectionDialog, OutputFolderDialog)
from ui.workers import (DrumSplitWorker, DrumPreloadWorker,
                        ChordAnalysisWorker, SeparationWorker,
                        RenderWorker, BeatAnalysisWorker)
from ui.timeline import (TimelineScene, TimelineView, TimeRuler,
                         ChordStrip, CHORD_HEIGHT, TrackHeader, TrackHeaderColumn, HEADER_WIDTH,
                         RULER_HEIGHT, time_to_x)
from ui.channel_strip import MixerPanel
from ui.widgets import make_icon, Stepper
from core.resources import (run_prerequisite_checks,
                            check_system_resources)
from core.dsp import DSPChain
from core.export import EXPORT_FORMATS, encode_with_ffmpeg
from core.audio_io import load_audio
from core import beats as beat_tools
from core.timeline_model import Clip, Track, Project
from core.timestretch import (render_clips, clip_needs_work,
                              available_backend, INSTALL_HINT)
from core.engine import TimelineEngine


STEM_NAMES = ('vocals', 'drums', 'bass', 'other', 'guitar', 'piano')
DRUM_PARTS = ('kick', 'snare', 'toms', 'hh', 'ride', 'crash')


def fmt_clock(t):
    t = max(0.0, t)
    return "{:02d}:{:04.1f}".format(int(t // 60), t % 60)


class RenderExportWorker(QThread):
    """Renders the timeline and encodes it off the UI thread."""
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, engine, path, fmt, bitrate, clips, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.path = path
        self.fmt = fmt
        self.bitrate = bitrate
        self.clips = clips

    def run(self):
        tmp_wav = self.path + ".tmp.wav"
        try:
            # Previews use the fast engine; export re-renders any
            # tempo/key-changed clip with the high-quality one.
            render_clips(self.clips, "high",
                         progress_cb=lambda p, m: self.progress.emit(
                             int(p * 0.4), "High-quality " + m.lower()))
            self.progress.emit(45, "Rendering mix...")
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

class StemExportWorker(QThread):
    """Renders each track as its own WAV, off the UI thread."""
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, engine, out_dir, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.out_dir = out_dir

    def run(self):
        try:
            files = export_stems(
                self.engine, self.out_dir,
                progress_cb=lambda p, m: self.progress.emit(p, m))
            self.finished_ok.emit(files)
        except Exception as e:
            self.failed.emit(str(e))


class DAWWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StemsMix")
        self.resize(1500, 960)
        self.setMinimumSize(1280, 700)
        self.history = History()
        self.project = Project()
        self.project_path = None
        self.dsp_chains = {}
        self.track_headers = {}
        self.selected_clip = None
        self.engine = None
        self.export_worker = None
        self.drum_worker = None
        self.drum_preload = None
        self.chord_worker = None
        self.sep_worker = None
        self.render_worker = None
        self.beat_worker = None
        self._beat_token = 0
        self.stem_paths = {}
        self._beat_flash = (False, False)
        self._render_dirty = False
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(350)   # coalesce rapid +/- clicks
        self._render_timer.timeout.connect(self._start_render)
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(40)
        self.playback_timer.timeout.connect(self.update_playhead)
        self.init_ui()
        for key, fn in ((Qt.Key.Key_Space, self.toggle_playback),
                        (Qt.Key.Key_Home, self.rewind)):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(fn)
        self._project_shortcuts = install_project_shortcuts(self)
        self.stem_export_worker = None
                  # <-- add this line inside __init__

    # ------------------------------------------------------------------ UI
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_transport())
        root.addWidget(self._build_songbar())

        top = QWidget()
        top_l = QVBoxLayout(top)
        top_l.setContentsMargins(0, 0, 0, 0)
        top_l.setSpacing(0)
        top_l.addWidget(self._build_arrange(), 1)
        top_l.addWidget(self._build_inspector())

        self.mixer = MixerPanel()
        self.mixer.mute_toggled.connect(self.set_track_mute)
        self.mixer.solo_toggled.connect(self.set_track_solo)
        self.mixer.volume_changed.connect(self.set_track_volume)
        self.mixer.pan_changed.connect(self.set_track_pan)
        self.mixer.master_changed.connect(self.set_master_volume)

        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        split.addWidget(top)
        split.addWidget(self.mixer)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([540, 340])
        self.mixer.setMinimumHeight(330)
        root.addWidget(split, 1)
        root.addWidget(self._build_statusbar())
        self._refresh_tool_buttons()
        # Attach the menu bar once, after every widget exists.
        from ui.menubar import build_menubar
        self._menubar = build_menubar(self)


    def import_stems(self):
        """Import additional audio files into the project."""
        from ui.import_stems_dialog import ImportStemsDialog
        from core.audio_io import load_audio
        from core.timeline_model import Clip, Track
        from core.dsp import DSPChain

        if not self.project.tracks:
            # Empty project: nothing to import into; load as a fresh set instead.
            self.status.setText(
                "Load a stems folder first, then import additional audio.")
            return

        playhead = self.engine.playhead_seconds if self.engine else 0.0
        dlg = ImportStemsDialog(self.project, playhead, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        files = dlg.selected_files()
        mode = dlg.import_mode()
        target = dlg.target_track_name()

        self._push_undo("Import stems")
        self._halt_playback()

        added_tracks = 0
        added_clips = 0
        failed = []

        for path in files:
            try:
                data, sr = load_audio(path)
            except Exception as e:
                failed.append((path, str(e)))
                continue

            name = os.path.splitext(os.path.basename(path))[0]

            if mode == "new_track":
                # Ensure unique track name
                base = name
                suffix = 1
                existing = {t.name for t in self.project.tracks}
                while name in existing:
                    suffix += 1
                    name = "{}_{}".format(base, suffix)
                track = Track(name)
                clip = Clip(name, path, data, sr, start_time=playhead)
                track.add_clip(clip)
                self.project.add_track(track)
                self.dsp_chains[name] = DSPChain(sr)
                added_tracks += 1
            else:
                # Add a clip to the target track
                track = next(
                    (t for t in self.project.tracks if t.name == target),
                    None)
                if track is None:
                    failed.append((path, "target track gone"))
                    continue
                clip = Clip(target, path, data, sr, start_time=playhead)
                track.add_clip(clip)
                added_clips += 1

        self.project._recompute_duration()
        self._rebuild_ui()

        summary = []
        if added_tracks:
            summary.append("{} new track(s)".format(added_tracks))
        if added_clips:
            summary.append("{} clip(s)".format(added_clips))
        if failed:
            summary.append("{} failed".format(len(failed)))
            for p, err in failed[:3]:
                print("Import failed for {}: {}".format(p, err))
        self.status.setText("Imported " + ", ".join(summary) + ".")

    # --- Menu-driven actions -------------------------------------------------

    def save_project(self, save_as=False):
        from ui.project_actions import save_project_dialog
        path = save_project_dialog(self, save_as=save_as)
        if path:
            self.project_path = path

    def open_project(self):
        from ui.project_actions import open_project_dialog
        result = open_project_dialog(self)
        if result is not None:
            self.project_path = result.path
            self._install_loaded_project(result)


    def open_batch(self):
        from ui.batch_dialog import BatchDialog
        BatchDialog(self).exec()

    def undo(self):
        if not getattr(self, "history", None) or not self.history.can_undo():
            self.status.setText("Nothing to undo.")
            return
        self._halt_playback()
        result = self.history.undo(self.project)
        if result is None:
            return
        label, restored = result
        self._install_project(restored)
        self.status.setText("Undo: " + label)

    def redo(self):
        if not getattr(self, "history", None) or not self.history.can_redo():
            self.status.setText("Nothing to redo.")
            return
        self._halt_playback()
        result = self.history.redo(self.project)
        if result is None:
            return
        label, restored = result
        self._install_project(restored)
        self.status.setText("Redo: " + label)

    def _build_transport(self):
        bar = QFrame()
        bar.setObjectName("transport")
        bar.setFixedHeight(58)
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 0, 16, 0)
        h.setSpacing(8)
        brand = QLabel("StemsMix")
        brand.setObjectName("brand")
        h.addWidget(brand)
        h.addSpacing(14)
        self.btn_open = QPushButton("Open Audio && Separate...")
        self.btn_open.setObjectName("primary")
        self.btn_open.clicked.connect(self.open_audio_file)
        self.btn_load = QPushButton("Load Stems...")
        self.btn_load.clicked.connect(self.load_stems)
        self.btn_save = QPushButton("Save")
        self.btn_save.setToolTip("Save project (Ctrl/Cmd + S)")
        self.btn_save.clicked.connect(
            lambda: save_project_dialog(self, save_as=False))
        self.btn_save_as = QPushButton("Save As…")
        self.btn_save_as.setToolTip("Save project to a new file (Ctrl/Cmd + Shift + S)")
        self.btn_save_as.clicked.connect(
            lambda: save_project_dialog(self, save_as=True))
        self.btn_open_project = QPushButton("Open Project…")
        self.btn_open_project.setToolTip("Open a .stemmix project (Ctrl/Cmd + O)")
        self.btn_open_project.clicked.connect(
            lambda: self._open_project_clicked())
        h.addWidget(self.btn_save)
        h.addWidget(self.btn_save_as)
        h.addWidget(self.btn_open_project)
        h.addWidget(self.btn_open)
        h.addWidget(self.btn_load)
        h.addStretch(1)

        self.btn_rewind = self._transport_btn("rewind", "Return to start (Home)")
        self.btn_play = self._transport_btn("play", "Play / Pause (Space)")
        self.btn_stop = self._transport_btn("stop", "Stop")
        self.btn_rewind.clicked.connect(self.rewind)
        self.btn_play.clicked.connect(self.toggle_playback)
        self.btn_stop.clicked.connect(self.stop_playback)
        for b in (self.btn_rewind, self.btn_play, self.btn_stop):
            h.addWidget(b)
        h.addSpacing(10)
        box = QFrame()
        box.setObjectName("timeBox")
        bl = QHBoxLayout(box)
        bl.setContentsMargins(14, 2, 14, 2)
        bl.setSpacing(8)
        self.lbl_time = QLabel(fmt_clock(0))
        self.lbl_time.setObjectName("timecode")
        self.lbl_total = QLabel("/ " + fmt_clock(0))
        self.lbl_total.setObjectName("timecodeDim")
        bl.addWidget(self.lbl_time)
        bl.addWidget(self.lbl_total)
        h.addWidget(box)
        h.addStretch(1)
        self.btn_export = QPushButton("Export Mix")
        self.btn_export.setObjectName("primary")
        self.btn_export.clicked.connect(self.export_mix)
        h.addWidget(self.btn_export)
        self.btn_export_stems = QPushButton("Export Stems")
        self.btn_export_stems.setObjectName("primary")
        self.btn_export_stems.setToolTip("Export each track as its own WAV")
        self.btn_export_stems.clicked.connect(self.export_stems)
        h.addWidget(self.btn_export_stems)
        self.btn_batch = QPushButton("Batch…")
        self.btn_batch.setToolTip("Separate many files in one run")
        self.btn_batch.clicked.connect(self.open_batch)
        h.addWidget(self.btn_batch)
        return bar


    def _build_songbar(self):
        bar = QFrame()
        bar.setObjectName("songbar")
        bar.setFixedHeight(48)
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 0, 16, 0)
        h.setSpacing(10)

        self.tempo_stepper = Stepper(
            "TEMPO", "Playback tempo, pitch preserved (double-click the "
            "value to reset)")
        self.tempo_stepper.stepped.connect(self.step_tempo)
        self.tempo_stepper.reset_requested.connect(
            lambda: self.change_tempo(1.0))
        self.key_stepper = Stepper(
            "KEY", "Transpose in semitones, tempo preserved (double-click "
            "the value to reset)")
        self.key_stepper.stepped.connect(self.step_key)
        self.key_stepper.reset_requested.connect(lambda: self.change_key(0))
        h.addWidget(self.tempo_stepper)
        h.addWidget(self.key_stepper)
        h.addSpacing(6)

        # detected tempo of the song (follows the Tempo control)
        box = QFrame()
        box.setObjectName("stepper")
        bl = QHBoxLayout(box)
        bl.setContentsMargins(10, 3, 6, 3)
        bl.setSpacing(4)
        t = QLabel("BPM")
        t.setObjectName("sectionLabel")
        self.lbl_bpm = QLabel("--")
        self.lbl_bpm.setObjectName("stepValue")
        self.lbl_bpm.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_bpm.setFixedWidth(60)
        self.btn_half = QPushButton("\u00f72")
        self.btn_half.setObjectName("tinyWide")
        self.btn_half.setToolTip("Detected tempo is double the real one")
        self.btn_half.clicked.connect(self.halve_bpm)
        self.btn_double = QPushButton("\u00d72")
        self.btn_double.setObjectName("tinyWide")
        self.btn_double.setToolTip("Detected tempo is half the real one")
        self.btn_double.clicked.connect(self.double_bpm)
        for w in (t, self.lbl_bpm, self.btn_half, self.btn_double):
            bl.addWidget(w)
        h.addWidget(box)

        self.btn_click = QPushButton(" Click")
        self.btn_click.setObjectName("clickButton")
        self.btn_click.setCheckable(True)
        self.btn_click.setIcon(make_icon("metronome", size=16))
        self.btn_click.setIconSize(QSize(16, 16))
        self.btn_click.setToolTip(
            "Metronome on the song's beat (follows the Tempo control; "
            "never included in exports)")
        self.btn_click.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_click.toggled.connect(self.toggle_click)
        self.beat_dot = QLabel()
        self.beat_dot.setObjectName("beatDot")
        self.beat_dot.setFixedSize(12, 12)
        self.beat_dot.setProperty("on", False)
        self.beat_dot_accent = QLabel()
        self.beat_dot_accent.setObjectName("beatDotAccent")
        self.beat_dot_accent.setFixedSize(12, 12)
        self.beat_dot_accent.setProperty("on", False)
        self.slider_click = QSlider(Qt.Orientation.Horizontal)
        self.slider_click.setRange(0, 100)
        self.slider_click.setValue(60)
        self.slider_click.setFixedWidth(80)
        self.slider_click.setToolTip("Click volume")
        self.slider_click.valueChanged.connect(self.set_click_volume)
        self.meter_stepper = Stepper(
            "METER", "Beats per bar (accented click on the first)")
        self.meter_stepper.stepped.connect(self.step_meter)
        self.down_stepper = Stepper(
            "DOWNBEAT", "Which detected beat is beat 1 of the bar. "
            "The automatic guess is often right; nudge it if not.")
        self.down_stepper.stepped.connect(self.step_downbeat)
        for w in (self.btn_click, self.beat_dot_accent, self.beat_dot,
                  self.slider_click, self.meter_stepper, self.down_stepper):
            h.addWidget(w)
        h.addStretch(1)

        self.btn_split_drums = QPushButton("Split Drums")
        self.btn_split_drums.clicked.connect(self.split_drums)
        self.btn_chords = QPushButton("Detect Chords")
        self.btn_chords.clicked.connect(self.detect_chords)
        h.addWidget(self.btn_split_drums)
        h.addWidget(self.btn_chords)
        return bar

    @staticmethod
    def _transport_btn(kind, tip):
        b = QPushButton()
        b.setObjectName("transport")
        b.setIcon(make_icon(kind))
        b.setIconSize(QSize(18, 18))
        b.setToolTip(tip)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return b

    def _build_arrange(self):
        self.scene = TimelineScene(num_tracks=6)
        self.view = TimelineView(self.scene)
        self.ruler = TimeRuler(self.view)
        self.chord_strip = ChordStrip(self.view)
        self.chord_strip.seek_requested.connect(self.on_seek)
        self.chords = []
        self.header_col = TrackHeaderColumn()
        self.scene.selectionChanged.connect(self.on_selection_changed)
        self.scene.clip_moved.connect(self.on_clip_moved)
        self.ruler.seek_requested.connect(self.on_seek)
        self.view.chrome_changed.connect(self.header_col.set_bottom_pad)
        self.header_col.scrolled.connect(
            self.view.verticalScrollBar().setValue)
        self.view.verticalScrollBar().valueChanged.connect(
            self.header_col.verticalScrollBar().setValue)

        corner = QFrame()
        corner.setObjectName("corner")
        corner.setFixedSize(HEADER_WIDTH, RULER_HEIGHT)
        cl = QHBoxLayout(corner)
        cl.setContentsMargins(12, 0, 8, 0)
        cl.setSpacing(4)
        lbl = QLabel("TRACKS")
        lbl.setObjectName("sectionLabel")
        cl.addWidget(lbl)
        cl.addStretch()
        zo = QPushButton()
        zo.setObjectName("tiny")
        zo.setIcon(make_icon("minus", size=14))
        zo.setToolTip("Zoom out (Ctrl/Cmd + scroll)")
        zi = QPushButton()
        zi.setObjectName("tiny")
        zi.setIcon(make_icon("plus", size=14))
        zi.setToolTip("Zoom in (Ctrl/Cmd + scroll)")
        fit = QPushButton("Fit")
        fit.setObjectName("tinyWide")
        fit.setToolTip("Fit the whole project in view")
        zo.clicked.connect(lambda: self.view.zoom_by(1 / 1.4))
        zi.clicked.connect(lambda: self.view.zoom_by(1.4))
        fit.clicked.connect(
            lambda: self.view.fit_to(self.project.total_duration))
        for b in (zo, zi, fit):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            cl.addWidget(b)

        w = QWidget()
        g = QGridLayout(w)
        g.setContentsMargins(0, 0, 0, 0)
        g.setSpacing(0)
        self.chord_label = QFrame()
        self.chord_label.setObjectName("corner")
        self.chord_label.setFixedSize(HEADER_WIDTH, CHORD_HEIGHT)
        ll = QHBoxLayout(self.chord_label)
        ll.setContentsMargins(12, 0, 8, 0)
        lbl2 = QLabel("CHORDS")
        lbl2.setObjectName("sectionLabel")
        ll.addWidget(lbl2)
        self.chord_label.setVisible(False)
        self.chord_strip.setVisible(False)
        g.addWidget(corner, 0, 0)
        g.addWidget(self.ruler, 0, 1)
        g.addWidget(self.chord_label, 1, 0)
        g.addWidget(self.chord_strip, 1, 1)
        g.addWidget(self.header_col, 2, 0)
        g.addWidget(self.view, 2, 1)
        g.setColumnStretch(1, 1)
        g.setRowStretch(2, 1)
        return w

    def _build_inspector(self):
        bar = QFrame()
        bar.setObjectName("inspector")
        bar.setFixedHeight(46)
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 0, 16, 0)
        h.setSpacing(10)
        tag = QLabel("CLIP")
        tag.setObjectName("sectionLabel")
        h.addWidget(tag)
        self.lbl_clip_name = QLabel("No clip selected")
        self.lbl_clip_name.setObjectName("clipName")
        self.lbl_clip_name.setMinimumWidth(130)
        h.addWidget(self.lbl_clip_name)
        h.addSpacing(8)


        def field(text, spin):
            lab = QLabel(text)
            lab.setObjectName("sectionLabel")
            h.addWidget(lab)
            h.addWidget(spin)

        self.spin_start = QDoubleSpinBox()
        self.spin_start.setRange(0, 3600)
        self.spin_start.setDecimals(2)
        self.spin_start.setSuffix(" s")
        self.spin_start.valueChanged.connect(self.on_clip_start_changed)
        self.spin_stretch = QDoubleSpinBox()
        self.spin_stretch.setRange(0.25, 4.0)
        self.spin_stretch.setSingleStep(0.05)
        self.spin_stretch.setDecimals(2)
        self.spin_stretch.setSuffix(" x")
        self.spin_stretch.setValue(1.0)
        self.spin_stretch.valueChanged.connect(self.on_clip_stretch_changed)
        self.spin_pitch = QDoubleSpinBox()
        self.spin_pitch.setRange(-24.0, 24.0)
        self.spin_pitch.setSingleStep(1.0)
        self.spin_pitch.setDecimals(1)
        self.spin_pitch.setSuffix(" st")
        self.spin_pitch.setValue(0.0)
        self.spin_pitch.valueChanged.connect(self.on_clip_pitch_changed)
        field("START", self.spin_start)
        field("TEMPO", self.spin_stretch)
        field("KEY", self.spin_pitch)

        lab = QLabel("GAIN")
        lab.setObjectName("sectionLabel")
        h.addWidget(lab)
        self.slider_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_gain.setRange(0, 200)
        self.slider_gain.setValue(100)
        self.slider_gain.setFixedWidth(180)
        self.slider_gain.valueChanged.connect(self.on_clip_gain_changed)
        h.addWidget(self.slider_gain)
        self.lbl_gain = QLabel("0.0 dB")
        self.lbl_gain.setObjectName("timecodeDim")
        self.lbl_gain.setMinimumWidth(60)
        h.addWidget(self.lbl_gain)
        h.addStretch()
        self._set_inspector_enabled(False)
        self.btn_clip_fx = QPushButton("FX…")
        self.btn_clip_fx.setToolTip("Effects for this clip only")
        self.btn_clip_fx.clicked.connect(self._open_clip_fx)
        self.btn_clip_fx.setEnabled(False)
        h.addWidget(self.btn_clip_fx)
        return bar

    def _open_clip_fx(self):
        if not self.selected_clip:
            return
        from ui.clip_effects_dialog import ClipEffectsDialog
        clip, _ = self.selected_clip
        self._push_undo("Clip effects")
        dlg = ClipEffectsDialog(clip, self)
        dlg.exec()
        self.status.setText("Clip effects updated.")

    def _build_statusbar(self):
        bar = QFrame()
        bar.setObjectName("statusbar")
        bar.setFixedHeight(28)
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 0, 16, 0)
        h.setSpacing(12)
        self.status = QLabel(
            "Open an audio file to separate, or load a stems folder.")
        self.status.setObjectName("dawStatus")
        h.addWidget(self.status, 1)
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("dawStatus")
        self.progress_label.setVisible(False)
        h.addWidget(self.progress_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedWidth(180)
        self.progress.setVisible(False)
        h.addWidget(self.progress)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self.cancel_busy_job)
        h.addWidget(self.btn_cancel)
        return bar

    def _set_inspector_enabled(self, on):
        for w in (self.spin_start, self.spin_stretch, self.spin_pitch,
                  self.slider_gain):
            w.setEnabled(on)

    def start_background_preload(self):
        """Warm drumsep weights so the first Split Drums is faster."""
        self.drum_preload = DrumPreloadWorker()

        def _ready(device):
            # Don't clobber an active job status line.
            if self.sep_worker and self.sep_worker.isRunning():
                return
            if self.drum_worker and self.drum_worker.isRunning():
                return
            self.status.setText(
                "Ready — drum model cached on {}.".format(device))

        self.drum_preload.finished_ok.connect(_ready)
        self.drum_preload.start()

    def cancel_busy_job(self):
        for w in (self.sep_worker, self.drum_worker):
            if w is not None and w.isRunning() and hasattr(w, "cancel"):
                w.cancel()
                self.status.setText("Cancelling…")
                self.progress_label.setText("Cancelling…")

    # ------------------------------------------------------------- loading
    def load_stems(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Stems Folder")
        if folder:
            self._load_folder(folder)

    def _set_busy(self, busy):
        self.progress.setVisible(busy)
        self.progress_label.setVisible(busy)
        self.btn_cancel.setVisible(busy)
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

        out_dlg = OutputFolderDialog(input_file, self)
        if out_dlg.exec() != QDialog.DialogCode.Accepted:
            return
        out_dir = out_dlg.output_dir()
        os.makedirs(out_dir, exist_ok=True)

        dlg = ModelSelectionDialog(resources, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        model = dlg.selected_model()
        providers = ["cpu"] if dlg.force_cpu() else "auto"
        chunked = dlg.use_chunked()

        self._halt_playback()
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
        self.sep_worker.finished.connect(lambda: self._set_busy(False))
        self.sep_worker.start()

    def on_sep_progress(self, pct, msg):
        self.progress.setValue(pct)
        self.progress_label.setText(msg)

    def _load_folder(self, folder):
        project = Project()
        chains = {}
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
            project.samplerate = sr
            track = Track(name, COLORS["track_colors"].get(
                name, COLORS["accent"]))
            track.add_clip(Clip(name, p, data, sr, start_time=0.0))
            project.add_track(track)
            chains[name] = DSPChain(sr)
            loaded += 1
        if loaded == 0:
            # Keep whatever was loaded before instead of wiping it.
            self.status.setText("No valid stems found in " + folder)
            return
        self._halt_playback()
        project.click_enabled = self.project.click_enabled
        project.click_volume = self.project.click_volume
        self.project = project
        self.dsp_chains = chains
        self.stem_paths = {n: os.path.join(folder, n + ".wav")
                           for n in chains}
        self.chords = []
        self.chord_label.setVisible(False)
        self.chord_strip.setVisible(False)
        self._rebuild_ui()
        self._start_beat_analysis()
        # Fit once the layout has its real size.
        QTimer.singleShot(0, lambda: self.view.fit_to(
            self.project.total_duration))
        self.status.setText(
            "Loaded {} stems from {}".format(loaded, folder))

    def _rebuild_ui(self):
        try:
            self.scene.selectionChanged.disconnect(
                self.on_selection_changed)
        except (TypeError, RuntimeError):
            pass
        self._halt_playback()
        self.selected_clip = None
        self.scene.clear_clips()
        self.header_col.clear()
        self.track_headers = {}
        tracks = self.project.tracks
        self.scene.set_extent(len(tracks), self.project.total_duration)
        for i, track in enumerate(tracks):
            color = COLORS["track_colors"].get(track.name, COLORS["accent"])
            for clip in track.clips:
                sc = self.scene.add_clip(
                    track.name, clip.data, clip.samplerate, i)
                sc.clip_ref = clip
                sc.set_stretch(clip.stretch)
                sc.setX(time_to_x(clip.start_time))
            header = TrackHeader(track.name, color)
            header.set_muted(track.muted)
            header.set_soloed(track.soloed)
            header.mute_toggled.connect(self.set_track_mute)
            header.solo_toggled.connect(self.set_track_solo)
            self.header_col.add(header)
            self.track_headers[track.name] = header
        self.mixer.set_tracks(
            tracks, lambda n: self.dsp_chains[n],
            self.project.master_volume)
        self.engine = TimelineEngine(
            self.project, lambda name: self.dsp_chains[name])
        self.lbl_total.setText(
            "/ " + fmt_clock(self.project.total_duration))
        self._refresh_song_controls()
        self._show_playhead(0.0)
        self._refresh_tool_buttons()
        self.scene.selectionChanged.connect(
            self.on_selection_changed)
        self.on_selection_changed()

    def _refresh_tool_buttons(self):
        names = {t.name for t in self.project.tracks}
        self.btn_split_drums.setEnabled("drums" in names)
        self.btn_chords.setEnabled(
            bool(names & {"other", "guitar", "piano", "vocals"}))
        has = bool(names)
        self.btn_export.setEnabled(has)
        for b in (self.btn_play, self.btn_stop, self.btn_rewind):
            b.setEnabled(has)

        self.btn_export_stems.setEnabled(has)

    # ---------------------------------------------------- track controls
    def _track(self, name):
        return next((t for t in self.project.tracks if t.name == name), None)

    def set_track_mute(self, name, muted):
        t = self._track(name)
        if t:
            t.muted = muted
            self._sync_track(t)

    def set_track_solo(self, name, soloed):
        t = self._track(name)
        if t:
            t.soloed = soloed
            self._sync_track(t)

    def _sync_track(self, t):
        """Header and mixer strip show the same mute/solo state."""
        h = self.track_headers.get(t.name)
        if h:
            h.set_muted(t.muted)
            h.set_soloed(t.soloed)
        self.mixer.sync_track(t)

    def set_track_volume(self, name, gain):
        t = self._track(name)
        if t:
            t.volume = gain

    def set_track_pan(self, name, pan):
        t = self._track(name)
        if t:
            t.pan = pan

    def set_master_volume(self, gain):
        self.project.master_volume = gain

    # ------------------------------------------------------- clip editing
    def on_selection_changed(self):
        items = [i for i in self.scene.selectedItems()
                 if getattr(i, "clip_ref", None) is not None]
        if not items:
            self.selected_clip = None
            self.lbl_clip_name.setText("No clip selected")
            self._set_inspector_enabled(False)
            if hasattr(self, "btn_clip_fx"):
                self.btn_clip_fx.setEnabled(False)
            return
        item = items[0]
        clip = item.clip_ref
        self.selected_clip = (clip, item)
        self.lbl_clip_name.setText(clip.stem_name.capitalize())
        self._set_inspector_enabled(True)
        if hasattr(self, "btn_clip_fx"):
            self.btn_clip_fx.setEnabled(True)
        for w, v in ((self.spin_start, clip.start_time),
                     (self.spin_stretch, clip.stretch),
                     (self.spin_pitch, clip.semitones)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self.slider_gain.blockSignals(True)
        self.slider_gain.setValue(int(round(clip.gain * 100)))
        self.slider_gain.blockSignals(False)
        self._update_gain_label(clip.gain)

    def on_clip_start_changed(self, value):
        if self.selected_clip:
            self._push_undo("Move clip")
            clip, item = self.selected_clip
            clip.start_time = value
            item.setX(time_to_x(value))
            self._project_changed()

    def on_clip_moved(self):
        """A clip was dragged: keep project length and Start box in sync."""
        self._project_changed()
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
            self._project_changed()
            self._request_render()

    def on_clip_pitch_changed(self, value):
        if self.selected_clip:
            self.selected_clip[0].semitones = value
            self._request_render()

    def on_clip_gain_changed(self, value):
        if self.selected_clip:
            gain = value / 100.0
            self.selected_clip[0].gain = gain
            self._update_gain_label(gain)

    def _update_gain_label(self, gain):
        self.lbl_gain.setText(
            "-inf" if gain <= 0.001
            else "{:+.1f} dB".format(20 * math.log10(gain)))

    def _project_changed(self):
        self.project._recompute_duration()
        self.scene.ensure_duration(self.project.total_duration)
        self.lbl_total.setText("/ " + fmt_clock(self.project.total_duration))

    # ------------------------------------------------------ beats / click
    def _start_beat_analysis(self):
        weights = {"drums": 1.0, "bass": 0.6, "other": 0.4,
                   "guitar": 0.3, "piano": 0.3}
        picks = [(self.stem_paths[n], w) for n, w in weights.items()
                 if n in self.stem_paths]
        if not picks:
            picks = [(p, 1.0) for p in self.stem_paths.values()]
        self._beat_token += 1
        token = self._beat_token
        self.lbl_bpm.setText("...")
        if not picks:
            return
        self.beat_worker = BeatAnalysisWorker(picks)
        self.beat_worker.finished_ok.connect(
            lambda r, tok=token: self.on_beats_ready(r, tok))
        self.beat_worker.failed.connect(
            lambda m: self.status.setText("Beat detection failed: " + m))
        self.beat_worker.start()

    def on_beats_ready(self, res, token):
        if token != self._beat_token:
            return                       # a newer project was loaded
        p = self.project
        if not res:
            p.beat_times = None
            self.lbl_bpm.setText("--")
            self._refresh_beat_ui()
            self.status.setText("No steady beat found in this song.")
            return
        p.beat_times = res["beats"]
        p.bpm = res["bpm"]
        p.beats_per_bar = 4
        p.downbeat_phase = res["phase"] % 4
        self._refresh_beat_ui()
        self.status.setText(
            "{:.1f} BPM detected{}. Turn on Click to hear it; use \u00f72 / "
            "\u00d72 or DOWNBEAT if it is off.".format(
                res["bpm"], "" if res["steady"] else " (tempo drifts)"))

    def _has_beats(self):
        b = self.project.beat_times
        return b is not None and len(b) > 1

    def _refresh_beat_ui(self):
        p = self.project
        has = self._has_beats()
        for w in (self.btn_click, self.slider_click, self.btn_half,
                  self.btn_double, self.meter_stepper, self.down_stepper):
            w.setEnabled(has)
        for w, fn in ((self.btn_click,
                       lambda: self.btn_click.setChecked(p.click_enabled)),
                      (self.slider_click,
                       lambda: self.slider_click.setValue(
                           int(round(p.click_volume * 100))))):
            w.blockSignals(True)
            fn()
            w.blockSignals(False)
        if has:
            base = beat_tools.bpm_from_beats(p.beat_times) or p.bpm
            self.lbl_bpm.setText("{:.1f}".format(base * p.tempo))
            self.meter_stepper.set_text("{}/4".format(p.beats_per_bar))
            self.down_stepper.set_text(str(p.downbeat_phase + 1))
        self._refresh_grid()

    def _refresh_grid(self):
        p = self.project
        if self._has_beats() and p.click_enabled:
            times = np.asarray(p.beat_times) / max(p.tempo, 1e-3)
            idx = np.arange(len(times))
            mask = ((idx - p.downbeat_phase) % max(1, p.beats_per_bar)) == 0
            self.scene.set_beats(times, mask)
        else:
            self.scene.set_beats(None)

    def toggle_click(self, on):
        self.project.click_enabled = bool(on)
        self._refresh_grid()

    def set_click_volume(self, v):
        self.project.click_volume = v / 100.0

    def step_meter(self, d):
        p = self.project
        p.beats_per_bar = int(max(2, min(12, p.beats_per_bar + d)))
        p.downbeat_phase %= p.beats_per_bar
        self._refresh_beat_ui()

    def step_downbeat(self, d):
        p = self.project
        p.downbeat_phase = (p.downbeat_phase + d) % max(1, p.beats_per_bar)
        self._refresh_beat_ui()

    def halve_bpm(self):
        p = self.project
        if self._has_beats():
            p.beat_times = beat_tools.halve(p.beat_times)
            p.downbeat_phase %= p.beats_per_bar
            self._refresh_beat_ui()

    def double_bpm(self):
        p = self.project
        if self._has_beats():
            p.beat_times = beat_tools.double(p.beat_times)
            p.downbeat_phase %= p.beats_per_bar
            self._refresh_beat_ui()

    def _flash_beat(self):
        """Light up the beat indicators while playing."""
        p = self.project
        if not (self._has_beats() and self.engine
                and self.engine.is_playing):
            on, acc = False, False
        else:
            t = self.engine.playhead_seconds
            times = np.asarray(p.beat_times) / max(p.tempo, 1e-3)
            i = int(np.searchsorted(times, t, side="right")) - 1
            near = i >= 0 and (t - times[i]) < 0.09
            acc = bool(near and (i - p.downbeat_phase) % p.beats_per_bar == 0)
            on = bool(near and not acc)
        if (on, acc) != self._beat_flash:
            self._beat_flash = (on, acc)
            for w, v in ((self.beat_dot, on), (self.beat_dot_accent, acc)):
                w.setProperty("on", v)
                w.style().unpolish(w)
                w.style().polish(w)

    # ------------------------------------------------------ tempo / key
    def _refresh_song_controls(self):
        self.tempo_stepper.set_text(
            "{:.0f}%".format(self.project.tempo * 100))
        k = int(round(self.project.key))
        self.key_stepper.set_text("0" if k == 0 else "{:+d}".format(k))
        self.chord_strip.set_view_params(self.project.tempo,
                                         self.project.key)
        has = bool(self.project.tracks)
        self.tempo_stepper.setEnabled(has)
        self.key_stepper.setEnabled(has)
        self._refresh_beat_ui()

    def _all_clips(self):
        return [c for t in self.project.tracks for c in t.clips]

    def step_tempo(self, direction):
        self.change_tempo(round(self.project.tempo + 0.05 * direction, 2))

    def step_key(self, direction):
        self.change_key(self.project.key + direction)

    def change_tempo(self, new):
        """Song tempo as a ratio of the original (pitch preserved).
        Every clip's tempo is scaled and its start time moved so the
        arrangement keeps its musical alignment."""
        if not self.project.tracks:
            return
        new = min(2.0, max(0.5, round(float(new), 4)))
        old = self.project.tempo
        if abs(new - old) < 1e-6:
            return
        if abs(new - 1.0) > 1e-6 and available_backend() is None:
            self.status.setText(INSTALL_HINT)
            return
        r = new / old
        clips = self._all_clips()
        if any(not (0.25 <= c.stretch * r <= 4.0) for c in clips):
            self.status.setText(
                "That tempo would push a clip outside its 0.25x-4x range.")
            return
        for c in clips:
            c.stretch *= r
            c.start_time /= r
        self.project.tempo = new
        for item in self.scene.clips:
            if item.clip_ref is not None:
                item.set_stretch(item.clip_ref.stretch)
                item.setX(time_to_x(item.clip_ref.start_time))
        self._project_changed()
        self._refresh_song_controls()
        self.on_selection_changed()
        self._request_render()

    def change_key(self, new):
        """Transpose the whole song by whole semitones (tempo preserved)."""
        if not self.project.tracks:
            return
        new = int(max(-12, min(12, round(new))))
        old = self.project.key
        if new == old:
            return
        if new != 0 and available_backend() is None:
            self.status.setText(INSTALL_HINT)
            return
        for c in self._all_clips():
            c.semitones += new - old
        self.project.key = new
        self._refresh_song_controls()
        self.on_selection_changed()
        self._request_render()

    def _render_busy(self):
        return (self._render_timer.isActive()
                or (self.render_worker is not None
                    and self.render_worker.isRunning()))

    def _request_render(self):
        """Clips with new tempo/key are silent until rendered. Rendering
        starts shortly after the last change, in the background."""
        self._halt_playback()
        self.status.setText("Applying tempo / key...")
        self._render_timer.start()

    def _start_render(self):
        if self.render_worker is not None and self.render_worker.isRunning():
            self._render_dirty = True
            return
        todo = [c for c in self._all_clips() if clip_needs_work(c, "fast")]
        if not todo:
            self._render_finished(0)
            return
        self._set_busy(True)
        self.progress.setValue(0)
        self.progress_label.setText("Rendering...")
        self.render_worker = RenderWorker(todo, "fast")
        self.render_worker.progress.connect(self.on_sep_progress)
        self.render_worker.finished_ok.connect(self._render_finished)
        self.render_worker.failed.connect(self._render_failed)
        self.render_worker.finished.connect(self._render_cleanup)
        self.render_worker.start()

    def _render_finished(self, n):
        self._project_changed()
        t = self.project.tempo
        k = int(round(self.project.key))
        self.status.setText(
            "Tempo {:.0f}%, key {:+d}: ready to play.".format(t * 100, k)
            if (abs(t - 1) > 1e-6 or k) else "Original tempo and key.")

    def _render_failed(self, msg):
        self.status.setText("Tempo/key render failed: " + msg)

    def _render_cleanup(self):
        self._set_busy(False)
        if self._render_dirty:
            self._render_dirty = False
            self._request_render()

    # ----------------------------------------------------------- transport
    def _show_playhead(self, t, follow=False):
        self.scene.set_playhead(t)
        self.ruler.set_playhead(t)
        self.chord_strip.set_playhead(t)
        self.lbl_time.setText(fmt_clock(t))
        if follow:
            self.view.follow(t)

    def on_seek(self, t):
        if not self.engine:
            return
        t = max(0.0, min(t, self.project.total_duration))
        self.engine.seek(t)
        self._show_playhead(t)

    def rewind(self):
        if not self.engine:
            return
        self.on_seek(0.0)
        self.view.horizontalScrollBar().setValue(0)

    def _halt_playback(self):
        """Stop audio and the UI timer (used before project changes)."""
        if self.engine:
            self.engine.stop()
        self.playback_timer.stop()
        self.btn_play.setIcon(make_icon("play"))
        self.mixer.reset_levels()
        self._flash_beat()

    def toggle_playback(self):
        if not self.engine or not self.project.tracks:
            return
        if self.engine.is_playing:
            self._halt_playback()
            return
        if self._render_busy():
            self.status.setText("Wait for the tempo/key render to finish.")
            return
        if not all(c.is_ready() for t in self.project.tracks
                   for c in t.clips):
            self.status.setText(
                "Some tracks are not rendered yet - change tempo/key again "
                "or reset it.")
            return
        if self.engine.playhead_seconds >= self.project.total_duration - 0.05:
            self.engine.seek(0.0)
        self.engine.start(from_seconds=self.engine.playhead_seconds)
        self.playback_timer.start()
        self.btn_play.setIcon(make_icon("pause"))

    def stop_playback(self):
        if not self.engine:
            return
        self._halt_playback()
        self.on_seek(0.0)
        self.view.horizontalScrollBar().setValue(0)

    def update_playhead(self):
        if not self.engine:
            return
        self._show_playhead(self.engine.playhead_seconds, follow=True)
        self.mixer.update_levels(self.engine.levels,
                                 self.engine.master_level)
        self._flash_beat()
        if not self.engine.is_playing:      # reached the end
            self._halt_playback()

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
        if self._render_busy():
            self.status.setText("Wait for the tempo/key render to finish.")
            return
        # Offline render reuses the per-track DSP chains, so playback
        # must not run at the same time.
        if self.engine.is_playing:
            self.toggle_playback()
        self.btn_export.setEnabled(False)
        self.btn_play.setEnabled(False)
        self._set_busy(True)
        self.progress.setValue(0)
        clips = [c for t in self.project.tracks for c in t.clips]
        self.export_worker = RenderExportWorker(
            self.engine, path, fmt, br, clips)
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
        self._set_busy(True)
        self.progress.setValue(0)
        self.progress_label.setText("Splitting drums...")
        self.status.setText("Splitting drums...")
        self.drum_worker = DrumSplitWorker(drums_path, out_dir)
        self.drum_worker.progress.connect(self.on_sep_progress)
        self.drum_worker.finished_ok.connect(
            lambda paths: self.on_drums_split(paths, out_dir))
        self.drum_worker.failed.connect(
            lambda e: self.status.setText("Drum split failed: " + e))
        def _drum_done():
            self._set_busy(False)
            self._refresh_tool_buttons()

        self.drum_worker.finished.connect(_drum_done)
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
                clip.stretch, clip.semitones, clip.gain = (
                    src.stretch, src.semitones, src.gain)
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
        if any(c.needs_render() for t in new_tracks for c in t.clips):
            self._request_render()   # parts must follow the song's tempo/key
        self.status.setText(
            "Drums split into {} parts: {}".format(
                len(new_tracks), ", ".join(t.name for t in new_tracks)))

    def detect_chords(self):
        # Mix the harmonic stems; the bass helps pin down the root.
        weights = {"other": 1.0, "guitar": 1.0, "piano": 1.0, "bass": 0.7}
        picks = [(t.clips[0].filepath, weights[t.name])
                 for t in self.project.tracks
                 if t.name in weights and t.clips]
        if not picks:
            picks = [(t.clips[0].filepath, 1.0)
                     for t in self.project.tracks
                     if t.name == "vocals" and t.clips]
        if not picks:
            self.status.setText("No harmonic stem available.")
            return
        self.btn_chords.setEnabled(False)
        self._set_busy(True)
        self.progress.setValue(0)
        self.status.setText("Detecting chords...")
        self.chord_worker = ChordAnalysisWorker(picks)
        self.chord_worker.progress.connect(self.on_sep_progress)
        self.chord_worker.finished_ok.connect(self.on_chords_detected)
        self.chord_worker.failed.connect(
            lambda e: self.status.setText("Chord detection failed: " + e))
        self.chord_worker.finished.connect(self._chords_finished)
        self.chord_worker.start()

    def _chords_finished(self):
        self._set_busy(False)
        self._refresh_tool_buttons()

    def on_chords_detected(self, chords):
        self.chords = chords
        shown = bool(chords)
        self.chord_label.setVisible(shown)
        self.chord_strip.setVisible(shown)
        self.chord_strip.set_chords(chords)
        self.chord_strip.set_view_params(self.project.tempo, self.project.key)
        if not chords:
            self.status.setText("No chords detected.")
            return
        names = []
        for _, _, n in chords:
            if n not in names:
                names.append(n)
        self.status.setText(
            "{} chords found ({}). Click a chord to jump to it.".format(
                len(chords), ", ".join(names[:8]) + (" ..." if len(names) > 8 else "")))

    def closeEvent(self, event):
        try:
            self.scene.selectionChanged.disconnect(
                self.on_selection_changed)
        except (TypeError, RuntimeError):
            pass
        if self.engine:
            self.engine.stop()
        self.playback_timer.stop()
        for w in (self.export_worker, self.drum_worker,
                  self.chord_worker, self.sep_worker,
                  self.render_worker, self.beat_worker,
                  self.stem_export_worker):
            if w is not None and w.isRunning():
                w.terminate()
                w.wait(2000)
        event.accept()

    def _open_project_clicked(self):
        """User clicked Open Project (or pressed Ctrl/Cmd + O)."""
        result = open_project_dialog(self)
        if result is None:
            return
        self._install_loaded_project(result)

    def _install_loaded_project(self, result):
        """Install a ProjectLoadResult produced by ui.project_actions."""
        project = result.project
        self._halt_playback()

        # Build a DSP chain per track and apply the saved parameters.
        chains = {}
        from core.dsp import DSPChain
        from core.project_io import apply_dsp_params
        for track in project.tracks:
            dsp = DSPChain(project.samplerate)
            apply_dsp_params(dsp, result.dsp_params.get(track.name, {}))
            chains[track.name] = dsp

        self.project = project
        self.dsp_chains = chains
        self.stem_paths = {
            track.name: track.clips[0].filepath
            for track in project.tracks
            if track.clips
        }

        self.chords = []
        self.chord_label.setVisible(False)
        self.chord_strip.setVisible(False)
        self._rebuild_ui()

        # Beat grid and click are stored in the project; no need to re-detect.
        if project.beat_times:
            self._refresh_beat_ui()

        # Stretched/pitch-shifted clips are silent until re-rendered.
        if any(c.needs_render() for t in project.tracks for c in t.clips):
            self._request_render()
            self.status.setText(
                "Opened {} ({} tracks). Rendering tempo/key changes…".format(
                    os.path.basename(result.path), len(project.tracks)))
        else:
            self.status.setText(
                "Opened {} ({} tracks).".format(
                    os.path.basename(result.path), len(project.tracks)))

        QTimer.singleShot(0, lambda: self.view.fit_to(project.total_duration))

        if result.missing:
            self.status.setText(
                self.status.text()
                + "  ({} clips missing audio)".format(len(result.missing)))

    def _open_project_for_shortcut(self):
        """Ctrl/Cmd+O target. Same as _open_project_clicked, kept separate so
        the shortcut can be rebound independently."""
        self._open_project_clicked()

    def _push_undo(self, label):
        """Call before mutating project state."""
        self.history.push(label, self.project)

    def undo(self):
        if not self.history.can_undo():
            self.status.setText("Nothing to undo.")
            return
        self._halt_playback()
        result = self.history.undo(self.project)
        if result is None:
            return
        label, restored = result
        self._install_project(restored)
        self.status.setText("Undo: " + label)

    def redo(self):
        if not self.history.can_redo():
            self.status.setText("Nothing to redo.")
            return
        self._halt_playback()
        result = self.history.redo(self.project)
        if result is None:
            return
        label, restored = result
        self._install_project(restored)
        self.status.setText("Redo: " + label)

    def _install_project(self, project):
        """Adopt a Project (from undo/redo) and rebuild the UI."""
        self.project = project
        for name in list(self.dsp_chains.keys()):
            if not any(t.name == name for t in project.tracks):
                del self.dsp_chains[name]
        from core.dsp import DSPChain
        for track in project.tracks:
            if track.name not in self.dsp_chains:
                self.dsp_chains[track.name] = DSPChain(project.samplerate)
        self._rebuild_ui()
        if any(c.needs_render() for t in project.tracks for c in t.clips):
            self._request_render()


    def export_stems(self):
        if not self.project.tracks:
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, "Choose folder for stems")
        if not out_dir:
            return
        if self._render_busy():
            self.status.setText("Wait for the tempo/key render to finish.")
            return
        if self.stem_export_worker and self.stem_export_worker.isRunning():
            self.status.setText("Stem export already running.")
            return
        if self.engine.is_playing:
            self.toggle_playback()

        self.btn_export_stems.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_play.setEnabled(False)
        self._set_busy(True)
        self.progress.setValue(0)
        self.progress_label.setText("Exporting stems…")

        self.stem_export_worker = StemExportWorker(self.engine, out_dir)
        self.stem_export_worker.progress.connect(self.on_sep_progress)
        self.stem_export_worker.finished_ok.connect(self._stems_exported)
        self.stem_export_worker.failed.connect(
            lambda m: self.status.setText("Stem export failed: " + m))
        self.stem_export_worker.finished.connect(self._stem_export_cleanup)
        self.stem_export_worker.start()

    def _stems_exported(self, files):
        if not files:
            self.status.setText("Stem export produced no files.")
            return
        folder = os.path.dirname(files[0][1])
        self.status.setText(
            "Exported {} stems to {}".format(len(files), folder))

    def _stem_export_cleanup(self):
        self._set_busy(False)
        self.btn_export_stems.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_play.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    results = run_prerequisite_checks()
    # Always show the dialog when anything is missing or suboptimal so
    # the user can one-click install pip/brew packages.
    if any(r.status in ("warn", "fail") for r in results):
        dlg = PrerequisiteDialog(results)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)
    w = DAWWindow()
    w.show()
    QTimer.singleShot(400, w.start_background_preload)
    sys.exit(app.exec())
