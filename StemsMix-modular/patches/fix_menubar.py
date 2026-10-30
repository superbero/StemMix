#!/usr/bin/env python3
"""fix_menubar_position.py — move the misplaced build_menubar call from
whatever line it landed on into DAWWindow.init_ui, at the end.

Removes every occurrence of the misplaced three-line block and inserts a
single correct copy at the end of init_ui, just before the class method
that follows it (import_stems or save_project).
"""
import os
import re
import shutil
import sys
import time

DAW = os.path.join("apps", "daw.py")


def backup(path):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst = path + ".bak." + stamp
    shutil.copy2(path, dst)
    print("backup:", dst)


def main():
    if not os.path.isfile(DAW):
        print("ERROR: run from the project root.")
        sys.exit(1)

    backup(DAW)
    src = open(DAW, "r", encoding="utf-8").read()

    # --- 1. Remove every misplaced copy of the block ------------------
    # The block can be indented with 0, 4, 8, or any whitespace.
    # It contains two lines:
    #   <indent>from ui.menubar import build_menubar
    #   <indent>self._menubar = build_menubar(self)
    pat = re.compile(
        r"^[ \t]*from ui\.menubar import build_menubar\n"
        r"^[ \t]*self\._menubar = build_menubar\(self\)\n",
        re.MULTILINE)
    src, n = pat.subn("", src)
    print("removed {} misplaced build_menubar block{}".format(
        n, "" if n == 1 else "s"))

    # --- 2. Find the end of init_ui -----------------------------------
    # init_ui starts with `    def init_ui(self):` and ends at the next
    # `    def ` at 4-space indent. Find that method's span, then insert
    # at the very end of it (before the following `def`).
    start_match = re.search(
        r"^    def init_ui\(self\):\n", src, re.MULTILINE)
    if not start_match:
        print("ERROR: could not find init_ui")
        sys.exit(1)

    start = start_match.end()
    # The method body is everything until the next `    def `
    next_def = re.search(
        r"^    def ", src[start:], re.MULTILINE)
    if not next_def:
        print("ERROR: could not find the method after init_ui")
        sys.exit(1)
    end = start + next_def.start()

    body = src[start:end]

    # Find the trailing blank line(s) — insert before them.
    trimmed = body.rstrip("\n")
    trailing = body[len(trimmed):]
    if not trailing:
        trailing = "\n"

    insert = (
        "\n"
        "        # Attach the menu bar once, after every widget exists.\n"
        "        from ui.menubar import build_menubar\n"
        "        self._menubar = build_menubar(self)\n"
    )

    new_body = trimmed + insert + trailing
    src = src[:start] + new_body + src[end:]

    # --- 3. Verify the class still parses -----------------------------
    try:
        compile(src, DAW, "exec")
    except SyntaxError as e:
        print("SYNTAX ERROR after edit:", e)
        print("Restoring from backup.")
        backups = sorted(
            f for f in os.listdir(os.path.dirname(DAW))
            if f.startswith("daw.py.bak."))
        if backups:
            shutil.copy2(
                os.path.join(os.path.dirname(DAW), backups[-1]),
                DAW)
            print("Restored", backups[-1])
        sys.exit(1)

    open(DAW, "w", encoding="utf-8").write(src)
    print("OK: build_menubar now lives inside init_ui")


if __name__ == "__main__":
    main()
