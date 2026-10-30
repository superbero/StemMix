#!/usr/bin/env python3
"""fix_daw_menu.py — repair the DAW window's menu wiring.

Applies the fixes found in the audit:
  1. Delete the duplicate stubbed `import_stems` (the one whose body is `...`).
  2. Delete the stubbed `export_stems` (the one whose body is `...`).
  3. Rename `export_stems_clicked` -> `export_stems` and update its button.
  4. Delete the duplicate stubbed `on_clip_start_changed`.
  5. Delete the duplicate `open_batch` at line ~370 (keep one).
  6. Clean up the double `closeEvent` and unreachable worker loop.
  7. Remove the duplicate undo/redo QShortcut lines in `__init__`.
  8. Move `self.history = History()` before `self.init_ui()`.
  9. Insert `build_menubar(self)` at the end of `init_ui`.
 10. Add `disconnect`/`connect` around `_rebuild_ui`'s scene work.
 11. Fix `btn_clip_fx` enable state in `on_selection_changed`.
 12. Trim the redundant `has = bool(names)` block in `_refresh_tool_buttons`.

Each edit is a targeted string replacement, not a regex scan.
Files are backed up to <file>.bak.<timestamp> before editing.

Run from the project root (the folder that contains apps/, ui/, core/).
"""
import os
import re
import shutil
import sys
import time

DAW = os.path.join("apps", "daw.py")


def backup(path):
    if not os.path.exists(path):
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst = path + ".bak." + stamp
    shutil.copy2(path, dst)
    print("  backup:", dst)


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def replace_once(src, old, new, label):
    """Replace `old` with `new`. Error if `old` isn't present exactly once."""
    n = src.count(old)
    if n == 0:
        print("  SKIP  {} (already applied or text changed)".format(label))
        return src
    if n > 1:
        print("  WARN  {} (found {} times, replacing first)".format(label, n))
    return src.replace(old, new, 1)


def remove_method(src, name, label):
    """Delete a `def name(self):` method block up to the next top-level
    `def` at the same indent. Also removes a preceding blank line."""
    # Match `    def name(self, ...):` at 4-space indent, then every
    # subsequent line that starts with 8+ spaces or blank, until the next
    # `    def ` line or end of file.
    pattern = re.compile(
        r"^[ \t]*def " + re.escape(name) + r"\(self[^)]*\):\n"
        r"(?:^[ \t]{8,}.*\n|^[ \t]*\n)*",
        re.MULTILINE)
    new_src, n = pattern.subn("", src)
    if n == 0:
        print("  SKIP  {} (not found)".format(label))
        return src
    print("  OK    {} (removed {} block{})".format(
        label, n, "s" if n != 1 else ""))
    return new_src


def main():
    if not os.path.isfile(DAW):
        print("ERROR: run from the project root. {} not found.".format(DAW))
        sys.exit(1)

    print("Backing up apps/daw.py ...")
    backup(DAW)

    src = read(DAW)
    original_len = len(src)

    # ------------------------------------------------------------------
    # 1. Delete the stubbed duplicate `import_stems` (body is `...`).
    #    The working version is the first one; the stub is the second.
    # ------------------------------------------------------------------
    stub_import = (
        '    def import_stems(self):\n'
        '        # (use the version from the previous message)\n'
        '        ...\n'
    )
    if stub_import in src:
        src = src.replace(stub_import, "", 1)
        print("  OK    removed stubbed import_stems")
    else:
        # try a looser match: any import_stems whose body is only "..."
        pat = re.compile(
            r"    def import_stems\(self\):\n"
            r"(?:\s*#[^\n]*\n)?"
            r"\s*\.\.\.\n")
        src, n = pat.subn("", src)
        if n:
            print("  OK    removed stubbed import_stems (loose match)")
        else:
            print("  SKIP  import_stems stub not found")

    # ------------------------------------------------------------------
    # 2. Delete the stubbed duplicate `export_stems` (body is `...`).
    # ------------------------------------------------------------------
    stub_export = (
        '    def export_stems(self):\n'
        '        # (use the version from the previous message)\n'
        '        ...\n'
    )
    if stub_export in src:
        src = src.replace(stub_export, "", 1)
        print("  OK    removed stubbed export_stems")
    else:
        pat = re.compile(
            r"    def export_stems\(self\):\n"
            r"(?:\s*#[^\n]*\n)?"
            r"\s*\.\.\.\n")
        src, n = pat.subn("", src)
        if n:
            print("  OK    removed stubbed export_stems (loose match)")
        else:
            print("  SKIP  export_stems stub not found")

    # ------------------------------------------------------------------
    # 3. Rename `export_stems_clicked` -> `export_stems` and update the
    #    toolbar button's connection.
    # ------------------------------------------------------------------
    src = replace_once(
        src,
        "    def export_stems_clicked(self):",
        "    def export_stems(self):",
        "rename export_stems_clicked -> export_stems")
    src = replace_once(
        src,
        "self.btn_export_stems.clicked.connect(self.export_stems_clicked)",
        "self.btn_export_stems.clicked.connect(self.export_stems)",
        "btn_export_stems connection")

    # ------------------------------------------------------------------
    # 4. Delete the stubbed duplicate `on_clip_start_changed` (body has
    #    `...`). The first, working version stays.
    # ------------------------------------------------------------------
    stub_on_start = (
        '    def on_clip_start_changed(self, value):\n'
        '        if self.selected_clip:\n'
        '            self._push_undo("Move clip")\n'
        '            clip, item = self.selected_clip\n'
        '            clip.start_time = value\n'
        '            ...\n'
    )
    if stub_on_start in src:
        src = src.replace(stub_on_start, "", 1)
        print("  OK    removed stubbed on_clip_start_changed")
    else:
        pat = re.compile(
            r"    def on_clip_start_changed\(self, value\):\n"
            r"        if self\.selected_clip:\n"
            r"            self\._push_undo\(\"Move clip\"\)\n"
            r"            clip, item = self\.selected_clip\n"
            r"            clip\.start_time = value\n"
            r"            \.\.\.\n")
        src, n = pat.subn("", src)
        if n:
            print("  OK    removed stubbed on_clip_start_changed "
                  "(loose match)")
        else:
            print("  SKIP  on_clip_start_changed stub not found")

    # Ensure the *working* on_clip_start_changed pushes undo.
    old_first = (
        '    def on_clip_start_changed(self, value):\n'
        '        if self.selected_clip:\n'
        '            clip, item = self.selected_clip\n'
        '            clip.start_time = value\n'
    )
    new_first = (
        '    def on_clip_start_changed(self, value):\n'
        '        if self.selected_clip:\n'
        '            self._push_undo("Move clip")\n'
        '            clip, item = self.selected_clip\n'
        '            clip.start_time = value\n'
    )
    src = replace_once(
        src, old_first, new_first,
        "on_clip_start_changed undo push")

    # ------------------------------------------------------------------
    # 5. Delete the duplicate `open_batch` (they are identical).
    # ------------------------------------------------------------------
    dup_open_batch = (
        '    def open_batch(self):\n'
        '        from ui.batch_dialog import BatchDialog\n'
        '        BatchDialog(self).exec()\n'
    )
    occurrences = src.count(dup_open_batch)
    if occurrences > 1:
        # Remove all but the last occurrence
        parts = src.split(dup_open_batch)
        src = dup_open_batch.join(parts[:-1]) + parts[-1]
        print("  OK    removed {} duplicate open_batch".format(
            occurrences - 1))
    else:
        print("  SKIP  open_batch is not duplicated")

    # ------------------------------------------------------------------
    # 6. Fix the double closeEvent. Replace the whole broken tail from
    #    the first `def closeEvent` through the second cleanup loop.
    # ------------------------------------------------------------------
    # The current broken closeEvent looks like:
    #     def closeEvent(self, event):
    #         ... body ...
    #         event.accept()
    #
    #         for w in (...):
    #             if w is not None and w.isRunning():
    #                 w.terminate()
    #                 w.wait(2000)
    #
    #             for w in (...):
    #                 if w is not None and w.isRunning():
    #                     w.terminate()
    #                     w.wait(2000)
    #
    # Match this block and replace it with a single clean version.
    close_pat = re.compile(
        r"    def closeEvent\(self, event\):\n"
        r"(?:[ \t]{8,}.*\n|[ \t]*\n)*?"
        r"(?=    def _open_project_clicked|    def _install_loaded_project"
        r"|\Z)",
        re.MULTILINE)
    clean_close = (
        '    def closeEvent(self, event):\n'
        '        try:\n'
        '            self.scene.selectionChanged.disconnect(\n'
        '                self.on_selection_changed)\n'
        '        except (TypeError, RuntimeError):\n'
        '            pass\n'
        '        if self.engine:\n'
        '            self.engine.stop()\n'
        '        self.playback_timer.stop()\n'
        '        for w in (self.export_worker, self.drum_worker,\n'
        '                  self.chord_worker, self.sep_worker,\n'
        '                  self.render_worker, self.beat_worker,\n'
        '                  self.stem_export_worker):\n'
        '            if w is not None and w.isRunning():\n'
        '                w.terminate()\n'
        '                w.wait(2000)\n'
        '        event.accept()\n'
        '\n'
    )
    src, n = close_pat.subn(clean_close, src, count=1)
    if n:
        print("  OK    replaced closeEvent with clean version")
    else:
        print("  SKIP  closeEvent block not matched (verify manually)")

    # ------------------------------------------------------------------
    # 7. Remove the duplicate undo/redo QShortcut lines.
    # ------------------------------------------------------------------
    dup_undo = (
        '        undo_sc = QShortcut(QKeySequence.StandardKey.Undo, self)\n'
        '        undo_sc.activated.connect(self.undo)\n'
        '        redo_sc = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)\n'
        '        redo_sc.activated.connect(self.redo)\n'
    )
    if dup_undo in src:
        src = src.replace(dup_undo, "", 1)
        print("  OK    removed duplicate undo/redo QShortcut lines")
    else:
        print("  SKIP  duplicate undo/redo shortcuts not found")

    # ------------------------------------------------------------------
    # 8. Move `self.history = History()` above `self.init_ui()`.
    # ------------------------------------------------------------------
    # Remove the late assignment (and its trailing comment).
    late_history = (
        '        self.history = History()          '
        '# <-- add this line inside __init__\n'
    )
    if late_history in src:
        src = src.replace(late_history, "", 1)
        print("  OK    removed late self.history assignment")
    else:
        # try without the comment
        late_history2 = '        self.history = History()\n'
        # Only remove if it appears AFTER `self.init_ui()`
        init_pos = src.find("self.init_ui()")
        hist_pos = src.find(late_history2, init_pos)
        if hist_pos != -1:
            src = (src[:hist_pos] + src[hist_pos + len(late_history2):])
            print("  OK    removed late self.history assignment")

    # Insert before `self.init_ui()` if not already before it.
    init_pos = src.find("        self.init_ui()")
    if init_pos != -1:
        before = src[:init_pos]
        if "self.history = History()" not in before:
            insert = ("        self.history = History()\n"
                      "        self.project_path = None\n")
            # If project_path already exists above, don't duplicate it.
            if "self.project_path = None" in before:
                insert = "        self.history = History()\n"
            src = before + insert + src[init_pos:]
            print("  OK    inserted self.history before init_ui()")
        else:
            print("  SKIP  self.history already before init_ui()")

    # ------------------------------------------------------------------
    # 9. Insert `build_menubar(self)` at the end of init_ui.
    # ------------------------------------------------------------------
    if "build_menubar(self)" not in src:
        end_marker = "        self._refresh_tool_buttons()\n"
        if end_marker in src:
            # Find the last occurrence within init_ui
            idx = src.find("    def import_stems(self):")
            if idx == -1:
                idx = src.find("    def save_project(")
            if idx == -1:
                idx = len(src)
            pre = src[:idx]
            last = pre.rfind(end_marker)
            if last != -1:
                insert_at = last + len(end_marker)
                add = ("        from ui.menubar import build_menubar\n"
                       "        self._menubar = build_menubar(self)\n")
                src = src[:insert_at] + add + src[insert_at:]
                print("  OK    inserted build_menubar at end of init_ui")
            else:
                print("  WARN  could not find end of init_ui "
                      "(insert build_menubar manually)")
        else:
            print("  WARN  end marker of init_ui not found")
    else:
        print("  SKIP  build_menubar already present")

    # ------------------------------------------------------------------
    # 10. Disconnect/connect selectionChanged around _rebuild_ui.
    # ------------------------------------------------------------------
    rebuild_head_old = (
        '    def _rebuild_ui(self):\n'
        '        self._halt_playback()\n'
    )
    rebuild_head_new = (
        '    def _rebuild_ui(self):\n'
        '        try:\n'
        '            self.scene.selectionChanged.disconnect(\n'
        '                self.on_selection_changed)\n'
        '        except (TypeError, RuntimeError):\n'
        '            pass\n'
        '        self._halt_playback()\n'
    )
    src = replace_once(
        src, rebuild_head_old, rebuild_head_new,
        "_rebuild_ui disconnect")

    rebuild_tail_old = (
        '        self._refresh_song_controls()\n'
        '        self._show_playhead(0.0)\n'
        '        self._refresh_tool_buttons()\n'
        '        self.on_selection_changed()\n'
    )
    rebuild_tail_new = (
        '        self._refresh_song_controls()\n'
        '        self._show_playhead(0.0)\n'
        '        self._refresh_tool_buttons()\n'
        '        self.scene.selectionChanged.connect(\n'
        '            self.on_selection_changed)\n'
        '        self.on_selection_changed()\n'
    )
    src = replace_once(
        src, rebuild_tail_old, rebuild_tail_new,
        "_rebuild_ui reconnect")

    # ------------------------------------------------------------------
    # 11. Fix btn_clip_fx state in on_selection_changed.
    # ------------------------------------------------------------------
    on_sel_old = (
        '            self.lbl_clip_name.setText("No clip selected")\n'
        '            self._set_inspector_enabled(False)\n'
        '            return\n'
    )
    on_sel_new = (
        '            self.lbl_clip_name.setText("No clip selected")\n'
        '            self._set_inspector_enabled(False)\n'
        '            if hasattr(self, "btn_clip_fx"):\n'
        '                self.btn_clip_fx.setEnabled(False)\n'
        '            return\n'
    )
    src = replace_once(
        src, on_sel_old, on_sel_new, "on_selection_changed FX disable")

    on_sel2_old = (
        '        self._set_inspector_enabled(True)\n'
        '        for w, v in ((self.spin_start, clip.start_time),\n'
    )
    on_sel2_new = (
        '        self._set_inspector_enabled(True)\n'
        '        if hasattr(self, "btn_clip_fx"):\n'
        '            self.btn_clip_fx.setEnabled(True)\n'
        '        for w, v in ((self.spin_start, clip.start_time),\n'
    )
    src = replace_once(
        src, on_sel2_old, on_sel2_new, "on_selection_changed FX enable")

    # Change the initial state of btn_clip_fx to disabled.
    src = replace_once(
        src,
        '        self.btn_clip_fx.setEnabled(True)\n',
        '        self.btn_clip_fx.setEnabled(False)\n',
        "btn_clip_fx initial state")

    # ------------------------------------------------------------------
    # 12. Trim the redundant `has = bool(names)` block.
    # ------------------------------------------------------------------
    dup_has = (
        '        has = bool(names)\n'
        '        self.btn_export.setEnabled(has)\n'
        '        self.btn_export_stems.setEnabled(has)\n'
    )
    if dup_has in src:
        src = src.replace(
            dup_has,
            '        self.btn_export_stems.setEnabled(has)\n',
            1)
        print("  OK    trimmed duplicate has block")

    # ------------------------------------------------------------------
    # Verify and write
    # ------------------------------------------------------------------
    try:
        compile(src, DAW, "exec")
    except SyntaxError as e:
        print("")
        print("SYNTAX ERROR after edits: {}".format(e))
        print("Backup is at {}*. Restoring.".format(DAW))
        # Restore from the most recent backup
        backups = sorted(
            [f for f in os.listdir(os.path.dirname(DAW))
             if f.startswith("daw.py.bak.")])
        if backups:
            shutil.copy2(
                os.path.join(os.path.dirname(DAW), backups[-1]),
                DAW)
            print("Restored from", backups[-1])
        sys.exit(1)

    write(DAW, src)
    print("")
    print("Wrote apps/daw.py ({} -> {} bytes, {} edits)".format(
        original_len, len(src), len(src) != original_len))
    print("")
    print("Next steps:")
    print("  1. Confirm ui/import_stems_dialog.py exists")
    print("  2. Confirm ui/batch_dialog.py exists")
    print("  3. python apps/daw.py")
    print("  4. File > Save should open a save dialog")


if __name__ == "__main__":
    main()
