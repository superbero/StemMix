#!/usr/bin/env python3
"""fix_follow_handler.py — add the missing _set_follow method to
DAWWindow.

feature_2_follow.py inserted it before `def closeEvent(self, event):`,
but that string had been rewritten by an earlier patch, so the insertion
was a no-op. The button exists and is connected; the target method does
not exist.
"""
import os
import re
import shutil
import sys
import time

DAW = "apps/daw.py"


def main():
    if not os.path.isfile(DAW):
        print("ERROR: run from project root")
        sys.exit(1)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(DAW, DAW + ".bak." + stamp)
    print("backup:", DAW + ".bak." + stamp)

    src = open(DAW).read()

    if re.search(r"^    def _set_follow\(self", src, re.MULTILINE):
        print("SKIP: _set_follow already defined")
        return

    # Anchor on any `    def closeEvent(self, event):` line, regardless of
    # what other methods sit around it.
    m = re.search(r"^    def closeEvent\(self, event\):\n",
                  src, re.MULTILINE)
    if not m:
        print("ERROR: could not find closeEvent to anchor the insertion")
        sys.exit(1)

    insert = (
        "    def _set_follow(self, on):\n"
        "        self._follow_playhead = bool(on)\n"
        "        if hasattr(self, \"_settings\"):\n"
        "            self._settings.setValue(\"follow_playhead\", on)\n"
        "        if hasattr(self, \"btn_follow\"):\n"
        "            self.btn_follow.setText(\n"
        "                \"Follow\" if on else \"Fixed\")\n"
        "\n"
    )
    src = src[:m.start()] + insert + src[m.start():]

    try:
        compile(src, DAW, "exec")
    except SyntaxError as e:
        print("SYNTAX ERROR:", e)
        print("Restoring backup")
        shutil.copy2(DAW + ".bak." + stamp, DAW)
        sys.exit(1)

    open(DAW, "w").write(src)
    print("Added _set_follow to DAWWindow")
    print("Run: python3 apps/daw.py")


if __name__ == "__main__":
    main()
