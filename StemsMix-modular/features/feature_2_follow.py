#!/usr/bin/env python3
"""fix_fade_array_truth.py — replace `playback_data() or []` with an
explicit None check, so numpy arrays don't raise ValueError on truth
testing.
"""
import os
import re
import shutil
import sys
import time


def main():
    path = "core/engine.py"
    if not os.path.isfile(path):
        print("ERROR: run from project root")
        sys.exit(1)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, path + ".bak." + stamp)
    print("backup:", path + ".bak." + stamp)

    src = open(path).read()
    original = src

    # Fix every occurrence of `len(X.playback_data() or [])`
    # Pattern: len(<something>.playback_data() or [])
    pat = re.compile(
        r"len\(\s*([A-Za-z_][A-Za-z0-9_]*)\.playback_data\(\)\s+or\s+\[\]\s*\)")

    def repl(m):
        var = m.group(1)
        return ("len(_pd) if (_pd := {}.playback_data()) is not None "
                "else 0".format(var))

    src, n = pat.subn(repl, src)
    print("replaced {} `playback_data() or []` occurrence{}".format(
        n, "" if n == 1 else "s"))

    # Also fix a variant that assigns to a variable first, e.g.
    #   data = clip.playback_data() or []
    pat2 = re.compile(
        r"^([ \t]*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        r"([A-Za-z_][A-Za-z0-9_]*)\.playback_data\(\)\s+or\s+\[\][ \t]*$",
        re.MULTILINE)

    def repl2(m):
        indent, var, clip = m.group(1), m.group(2), m.group(3)
        return ("{0}{1} = {2}.playback_data()\n"
                "{0}if {1} is None:\n"
                "{0}    {1} = []".format(indent, var, clip))

    src, n2 = pat2.subn(repl2, src)
    print("replaced {} assignment-style occurrence{}".format(
        n2, "" if n2 == 1 else "s"))

    if src == original:
        print("Nothing to change — the pattern may already be fixed.")
        return

    try:
        compile(src, path, "exec")
    except SyntaxError as e:
        print("SYNTAX ERROR:", e)
        print("Restoring backup")
        shutil.copy2(path + ".bak." + stamp, path)
        sys.exit(1)

    open(path, "w").write(src)
    print("Fixed", path)


if __name__ == "__main__":
    main()
