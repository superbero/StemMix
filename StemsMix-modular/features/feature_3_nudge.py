#!/usr/bin/env python3
"""feature_3_nudge.py — keyboard nudge and beat quantize for the
selected clip.

Shortcuts (when a clip is selected):
  [     nudge left 10 ms
  ]     nudge right 10 ms
  {     nudge left 100 ms
  }     nudge right 100 ms
  Shift+[  nudge left 1 s
  Shift+]  nudge right 1 s
  N     quantize to nearest beat
  Shift+N  quantize to previous beat
  Alt+N    quantize to next beat
"""
import os
import shutil
import sys
import time


NUDGE_PY = '''"""Nudge and quantize helpers for the selected clip."""
import math


def quantize_time(t, beats, mode="nearest"):
    if not beats:
        return t
    import bisect
    i = bisect.bisect_left(beats, t)
    if i == 0:
        return beats[0]
    if i >= len(beats):
        return beats[-1]
    if mode == "floor":
        return beats[i - 1]
    if mode == "ceil":
        return beats[i]
    return beats[i] if abs(beats[i] - t) < abs(t - beats[i - 1]) else beats[i - 1]


def nudge_seconds(t, seconds, min_time=0.0):
    return max(min_time, t + seconds)


def quantize_by_fraction(t, bpm, fraction=1.0, mode="nearest"):
    """Snap t to a fraction of a beat. fraction=1 -> beats, 0.5 -> 8ths."""
    if bpm is None or bpm <= 0:
        return t
    step = (60.0 / bpm) * fraction
    if step <= 0:
        return t
    if mode == "floor":
        return math.floor(t / step) * step
    if mode == "ceil":
        return math.ceil(t / step) * step
    return round(t / step) * step
'''


def backup(path):
    if not os.path.exists(path):
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst = path + ".bak." + stamp
    shutil.copy2(path, dst)
    print("  backup:", dst)


def main():
    if not os.path.isdir("core") or not os.path.isdir("apps"):
        print("ERROR: run from project root")
        sys.exit(1)

    # 1. Write core/nudge.py
    backup("core/nudge.py")
    os.makedirs("core", exist_ok=True)
    open("core/nudge.py", "w").write(NUDGE_PY)
    print("  wrote core/nudge.py")

    # 2. Patch apps/daw.py
    path = "apps/daw.py"
    backup(path)
    src = open(path).read()

    if "_install_nudge_shortcuts" in src:
        print("  SKIP daw.py (already patched)")
    else:
        # Install shortcuts at end of __init__
        src = src.replace(
            "        self._project_shortcuts = install_project_shortcuts(self)\n",
            "        self._project_shortcuts = install_project_shortcuts(self)\n"
            "        self._install_nudge_shortcuts()\n",
            1)

        # Add the installer method before closeEvent
        installer = '''
    def _install_nudge_shortcuts(self):
        """Nudge and quantize shortcuts for the selected clip."""
        from PyQt6.QtGui import QKeySequence, QShortcut

        def do_nudge(seconds):
            if not self.selected_clip:
                return
            from core.nudge import nudge_seconds
            self._push_undo("Nudge clip")
            clip, item = self.selected_clip
            clip.start_time = nudge_seconds(clip.start_time, seconds)
            item.setX(time_to_x(clip.start_time))
            self._project_changed()
            self.status.setText(
                "Clip start: {:.3f}s".format(clip.start_time))

        def do_quantize(mode):
            if not self.selected_clip:
                return
            if not self._has_beats():
                self.status.setText("No beat grid to quantize against.")
                return
            from core.nudge import quantize_by_fraction
            from core.beats import bpm_from_beats
            self._push_undo("Quantize clip")
            clip, item = self.selected_clip
            bpm = None
            try:
                bpm = bpm_from_beats(self.project.beat_times)
            except Exception:
                pass
            eff = (bpm * self.project.tempo) if bpm else None
            clip.start_time = quantize_by_fraction(
                clip.start_time, eff, fraction=1.0, mode=mode)
            item.setX(time_to_x(clip.start_time))
            self._project_changed()
            self.status.setText(
                "Quantized to {:.3f}s".format(clip.start_time))

        self._nudge_shortcuts = []
        for seq, fn in (
                ("[", lambda: do_nudge(-0.010)),
                ("]", lambda: do_nudge(+0.010)),
                ("{", lambda: do_nudge(-0.100)),
                ("}", lambda: do_nudge(+0.100)),
                ("Shift+[", lambda: do_nudge(-1.000)),
                ("Shift+]", lambda: do_nudge(+1.000)),
                ("N", lambda: do_quantize("nearest")),
                ("Shift+N", lambda: do_quantize("floor")),
                ("Alt+N", lambda: do_quantize("ceil"))):
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(fn)
            self._nudge_shortcuts.append(sc)

    def closeEvent(self, event):
'''
        src = src.replace("    def closeEvent(self, event):\n",
                          installer, 1)

        try:
            compile(src, path, "exec")
        except SyntaxError as e:
            print("SYNTAX ERROR:", e)
            sys.exit(1)
        open(path, "w").write(src)
        print("  patched apps/daw.py")

    print("Feature 3 (nudge / quantize) applied.")
    print("Test: select a clip, press [ ] to nudge, N to quantize.")


if __name__ == "__main__":
    main()
