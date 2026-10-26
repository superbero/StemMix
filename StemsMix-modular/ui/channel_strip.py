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
