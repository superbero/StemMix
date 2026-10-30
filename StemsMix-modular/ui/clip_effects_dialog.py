"""Dialog for editing per-clip effects. Same knobs as a channel strip,
but stored on the clip rather than the track."""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QDialogButtonBox)
from PyQt6.QtCore import Qt

from .style import COLORS
from .widgets import Knob
from core.dsp import DSPChain


def _comp_fmt(v):
    return "Off" if v < 0.5 else "thr -{:.0f} dB".format(v)


class ClipEffectsDialog(QDialog):
    """Edits a DSPChain attached to a single clip. The clip has one only
    after the user changes at least one knob; cancel leaves it untouched."""

    def __init__(self, clip, parent=None):
        super().__init__(parent)
        self.clip = clip
        self.setWindowTitle("Clip Effects — " + clip.stem_name.capitalize())
        self.setMinimumWidth(420)

        if clip.effects is None:
            clip.effects = DSPChain(clip.samplerate)
        self.dsp = clip.effects

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("Clip Effects")
        title.setObjectName("header_title")
        layout.addWidget(title)

        hint = QLabel("These apply to this clip only, after the track's "
                      "channel strip.")
        hint.setWordWrap(True)
        hint.setStyleSheet(
            "color: " + COLORS["text_dim"] + "; font-size: 12px;")
        layout.addWidget(hint)

        def knob_row(knobs):
            row = QHBoxLayout()
            row.setSpacing(20)
            row.addStretch()
            for k in knobs:
                row.addWidget(k)
            row.addStretch()
            return row

        def make(label, lo, hi, attr, scale=1.0, fmt="{:+.0f} dB",
                 default=0.0, bipolar=True, eq=False):
            k = Knob(label, lo, hi, getattr(self.dsp, attr) * scale,
                     default, fmt, COLORS["accent"], 34, bipolar)

            def on_change(v):
                setattr(self.dsp, attr, v / scale)
                if eq:
                    self.dsp._filters_dirty = True
            k.valueChanged.connect(on_change)
            return k

        layout.addLayout(knob_row([
            make("LO", -12, 12, "eq_bass", eq=True),
            make("MID", -12, 12, "eq_mid", eq=True),
            make("HI", -12, 12, "eq_treble", eq=True),
        ]))

        layout.addLayout(knob_row([
            make("COMP", 0, 40, "comp_threshold", scale=-1.0,
                 fmt=_comp_fmt, default=0.0, bipolar=False),
            make("REV", 0, 100, "reverb_mix", scale=100.0,
                 fmt="{:.0f}%", default=0.0, bipolar=False),
        ]))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok:
            ok.setObjectName("primary")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        reset_btn = buttons.addButton("Reset All",
                                       QDialogButtonBox.ButtonRole.ResetRole)
        reset_btn.clicked.connect(self._reset_all)

    def _reset_all(self):
        for attr in ("eq_bass", "eq_mid", "eq_treble",
                     "comp_threshold", "reverb_mix"):
            setattr(self.dsp, attr, 0.0)
        self.dsp._filters_dirty = True
        self.close()
        self.__init__(self.clip, self.parent())
        self.show()
