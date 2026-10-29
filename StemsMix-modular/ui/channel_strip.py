"""Mixer: channel strips (EQ, dynamics, reverb, pan, fader, meter)."""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                             QScrollArea, QVBoxLayout, QWidget)

from .style import COLORS
from .widgets import Fader, Knob, LevelMeter, fmt_db


def _pan_fmt(v):
    if abs(v) < 0.01:
        return "C"
    return "L{}".format(round(-v * 100)) if v < 0 else "R{}".format(round(v * 100))


def _comp_fmt(v):
    """Knob value is the amount (0 = off); shown as the threshold."""
    return "Off" if v < 0.5 else "thr -{:.0f} dB".format(v)


class _StripBase(QFrame):
    WIDTH = 120
    KNOB_ROW = 42

    def __init__(self, title, color, parent=None):
        super().__init__(parent)
        self.setObjectName("strip")
        self.setFixedWidth(self.WIDTH)
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(0, 0, 0, 8)
        self.root.setSpacing(6)
        head = QLabel(title)
        head.setAlignment(Qt.AlignmentFlag.AlignCenter)
        head.setFixedHeight(24)
        head.setStyleSheet(
            "QLabel { background-color: %s; color: #101114; "
            "font-weight: 700; font-size: 12px; "
            "border-top-left-radius: 7px; border-top-right-radius: 7px; }"
            % color)
        self.root.addWidget(head)
        self.body = QVBoxLayout()
        self.body.setContentsMargins(6, 0, 6, 0)
        self.body.setSpacing(6)
        self.root.addLayout(self.body, 1)

    def _fader_block(self, gain):
        self.fader = Fader()
        self.fader.set_gain(gain)
        self.meter = LevelMeter()
        row = QHBoxLayout()
        row.setSpacing(2)
        row.addStretch()
        row.addWidget(self.fader)
        row.addWidget(self.meter)
        row.addStretch()
        self.body.addLayout(row, 1)
        self.readout = QLabel(fmt_db(gain))
        self.readout.setObjectName("dbReadout")
        self.readout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.body.addWidget(self.readout)
        self.fader.valueChanged.connect(
            lambda g: self.readout.setText(fmt_db(g)))

    def set_levels(self, l, r):
        self.meter.set_levels(l, r)

    def reset_levels(self):
        self.meter.reset()


class ChannelStrip(_StripBase):
    mute_toggled = pyqtSignal(str, bool)
    solo_toggled = pyqtSignal(str, bool)
    volume_changed = pyqtSignal(str, float)
    pan_changed = pyqtSignal(str, float)

    def __init__(self, track, dsp, parent=None):
        color = COLORS["track_colors"].get(track.name, COLORS["accent"])
        super().__init__(track.name.capitalize(), color, parent)
        self.name = track.name
        self.dsp = dsp

        def dsp_knob(label, lo, hi, attr, scale=1.0, fmt="{:+.0f} dB",
                     default=0.0, bipolar=True, eq=False):
            k = Knob(label, lo, hi, getattr(dsp, attr) * scale, default,
                     fmt, color, 28, bipolar)

            def on_change(v):
                setattr(dsp, attr, v / scale)
                if eq:
                    dsp._filters_dirty = True
            k.valueChanged.connect(on_change)
            return k

        eq_row = QHBoxLayout()
        eq_row.setSpacing(0)
        for k in (dsp_knob("LO", -12, 12, "eq_bass", eq=True),
                  dsp_knob("MID", -12, 12, "eq_mid", eq=True),
                  dsp_knob("HI", -12, 12, "eq_treble", eq=True)):
            eq_row.addWidget(k)
        self.body.addLayout(eq_row)

        fx_row = QHBoxLayout()
        fx_row.setSpacing(0)
        fx_row.addWidget(dsp_knob("COMP", 0, 40, "comp_threshold",
                                  scale=-1.0, fmt=_comp_fmt, default=0.0,
                                  bipolar=False))
        fx_row.addWidget(dsp_knob("REV", 0, 100, "reverb_mix", scale=100.0,
                                  fmt="{:.0f}%", default=0.0,
                                  bipolar=False))
        self.pan_knob = Knob("PAN", -1, 1, track.pan, 0.0, _pan_fmt,
                             color, 28, True)
        self.pan_knob.valueChanged.connect(
            lambda v: self.pan_changed.emit(self.name, v))
        fx_row.addWidget(self.pan_knob)
        self.body.addLayout(fx_row)

        ms = QHBoxLayout()
        ms.setSpacing(4)
        ms.addStretch()
        self.mute_btn = self._ms("M", "mute", track.muted)
        self.solo_btn = self._ms("S", "solo", track.soloed)
        self.mute_btn.toggled.connect(
            lambda c: self.mute_toggled.emit(self.name, c))
        self.solo_btn.toggled.connect(
            lambda c: self.solo_toggled.emit(self.name, c))
        ms.addWidget(self.mute_btn)
        ms.addWidget(self.solo_btn)
        ms.addStretch()
        self.body.addLayout(ms)

        self._fader_block(track.volume)
        self.fader.valueChanged.connect(
            lambda g: self.volume_changed.emit(self.name, g))

    @staticmethod
    def _ms(text, kind, checked):
        b = QPushButton(text)
        b.setObjectName("msButton")
        b.setProperty("kind", kind)
        b.setCheckable(True)
        b.setChecked(checked)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return b

    def set_muted(self, v):
        self.mute_btn.blockSignals(True)
        self.mute_btn.setChecked(v)
        self.mute_btn.blockSignals(False)

    def set_soloed(self, v):
        self.solo_btn.blockSignals(True)
        self.solo_btn.setChecked(v)
        self.solo_btn.blockSignals(False)


class MasterStrip(_StripBase):
    volume_changed = pyqtSignal(float)

    def __init__(self, gain=1.0, parent=None):
        super().__init__("Master", "#a7acb8", parent)
        # Same height as the knob/mute rows of a channel strip, so the
        # master fader scale lines up with the others.
        out = QLabel("OUTPUT")
        out.setObjectName("sectionLabel")
        out.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.body.addWidget(out)
        self.body.addSpacing(2 * self.KNOB_ROW + 20 + 2 * 6 - 14)
        self._fader_block(gain)
        self.fader.valueChanged.connect(self.volume_changed.emit)


class MixerPanel(QFrame):
    mute_toggled = pyqtSignal(str, bool)
    solo_toggled = pyqtSignal(str, bool)
    volume_changed = pyqtSignal(str, float)
    pan_changed = pyqtSignal(str, float)
    master_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("mixer")
        self.strips = {}
        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.holder = QWidget()
        self.holder.setObjectName("mixerHolder")
        self._lay = QHBoxLayout(self.holder)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(6)
        self._lay.addStretch()
        self.scroll.setWidget(self.holder)
        outer.addWidget(self.scroll, 1)
        self.master = MasterStrip()
        self.master.volume_changed.connect(self.master_changed.emit)
        outer.addWidget(self.master)

    def set_tracks(self, tracks, dsp_lookup, master_volume=1.0):
        while self._lay.count() > 1:
            item = self._lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.strips = {}
        for track in tracks:
            s = ChannelStrip(track, dsp_lookup(track.name))
            s.mute_toggled.connect(self.mute_toggled.emit)
            s.solo_toggled.connect(self.solo_toggled.emit)
            s.volume_changed.connect(self.volume_changed.emit)
            s.pan_changed.connect(self.pan_changed.emit)
            self._lay.insertWidget(self._lay.count() - 1, s)
            self.strips[track.name] = s
        self.master.fader.set_gain(master_volume)
        self.master.readout.setText(fmt_db(master_volume))

    def sync_track(self, track):
        s = self.strips.get(track.name)
        if s:
            s.set_muted(track.muted)
            s.set_soloed(track.soloed)

    def update_levels(self, levels, master):
        for name, s in self.strips.items():
            l, r = levels.get(name, (0.0, 0.0))
            s.set_levels(l, r)
        self.master.set_levels(*master)

    def reset_levels(self):
        for s in self.strips.values():
            s.reset_levels()
        self.master.reset_levels()
