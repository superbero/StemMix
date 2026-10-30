"""Track effects dialog.

Edits the DSPChain that lives on a Track. This replaces the earlier
per-clip effects design: effects belong to tracks (as in Logic, Ableton,
Reaper), and the DSPChain on each track is already what the engine
applies during playback and export.

Changes are applied live as the user drags a knob, so the dialog can
stay open while the transport plays. Cancel does not undo live changes
— that's what undo is for. The DAW pushes an undo snapshot before
opening the dialog.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QDialogButtonBox, QFrame)

from .style import COLORS
from .widgets import Knob


def _comp_fmt(v):
    return "Off" if v < 0.5 else "thr -{:.0f} dB".format(v)


class TrackFxDialog(QDialog):
    """Edits a Track's DSPChain in place. Uses the same knob widgets as
    the mixer's channel strips so a user sees identical controls in both
    places."""

    def __init__(self, track, dsp, parent=None):
        super().__init__(parent)
        self.track = track
        self.dsp = dsp
        self.setWindowTitle(
            "Effects — " + track.name.capitalize())
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel(track.name.capitalize())
        title.setObjectName("header_title")
        layout.addWidget(title)

        hint = QLabel(
            "Effects are applied on top of the clip's tempo and key, and "
            "are included in the mix and stem exports. Changes take "
            "effect immediately.")
        hint.setWordWrap(True)
        hint.setStyleSheet(
            "color: " + COLORS["text_dim"] + "; font-size: 12px;")
        layout.addWidget(hint)

        # --- EQ row ---
        eq_frame = QFrame()
        eq_frame.setObjectName("fxGroup")
        eq_row = QHBoxLayout(eq_frame)
        eq_row.setContentsMargins(12, 12, 12, 12)
        eq_row.setSpacing(24)
        self.eq_bass = self._knob(eq_row, "LO", -12, 12, "eq_bass",
                                  eq=True)
        self.eq_mid = self._knob(eq_row, "MID", -12, 12, "eq_mid",
                                 eq=True)
        self.eq_treble = self._knob(eq_row, "HI", -12, 12, "eq_treble",
                                    eq=True)
        self._group_label(layout, "EQ", eq_frame)

        # --- Dynamics row ---
        dyn_frame = QFrame()
        dyn_frame.setObjectName("fxGroup")
        dyn_row = QHBoxLayout(dyn_frame)
        dyn_row.setContentsMargins(12, 12, 12, 12)
        dyn_row.setSpacing(24)
        self.comp = self._knob(dyn_row, "COMP", 0, 40, "comp_threshold",
                               scale=-1.0, fmt=_comp_fmt, bipolar=False)
        self.reverb = self._knob(dyn_row, "REV", 0, 100, "reverb_mix",
                                 scale=100.0, fmt="{:.0f}%",
                                 bipolar=False)
        self._group_label(layout, "Dynamics", dyn_frame)

        # --- Buttons ---
        buttons = QDialogButtonBox()
        reset = buttons.addButton(
            "Reset All", QDialogButtonBox.ButtonRole.ResetRole)
        reset.clicked.connect(self._reset_all)
        close = buttons.addButton(
            "Close", QDialogButtonBox.ButtonRole.AcceptRole)
        close.setObjectName("primary")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.accept)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------

    def _group_label(self, layout, text, frame):
        lab = QLabel(text.upper())
        lab.setObjectName("sectionLabel")
        layout.addWidget(lab)
        layout.addWidget(frame)

    def _knob(self, row, label, lo, hi, attr, scale=1.0,
              fmt="{:+.0f} dB", bipolar=True, eq=False):
        k = Knob(label, lo, hi, getattr(self.dsp, attr) * scale,
                 default=0.0, fmt=fmt, color=COLORS["accent"],
                 diameter=38, bipolar=bipolar)

        def on_change(v):
            setattr(self.dsp, attr, v / scale)
            if eq:
                self.dsp._filters_dirty = True
        k.valueChanged.connect(on_change)
        row.addWidget(k)
        return k

    def _reset_all(self):
        for attr in ("eq_bass", "eq_mid", "eq_treble",
                     "comp_threshold", "reverb_mix"):
            setattr(self.dsp, attr, 0.0)
        self.dsp._filters_dirty = True
        # Reset the visible knobs too, without re-emitting change events
        for knob, attr, scale in (
                (self.eq_bass, "eq_bass", 1.0),
                (self.eq_mid, "eq_mid", 1.0),
                (self.eq_treble, "eq_treble", 1.0),
                (self.comp, "comp_threshold", -1.0),
                (self.reverb, "reverb_mix", 100.0)):
            knob.blockSignals(True)
            knob.set_value(getattr(self.dsp, attr) * scale)
            knob.blockSignals(False)
