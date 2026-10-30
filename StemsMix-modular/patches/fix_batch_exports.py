#!/usr/bin/env python3
"""create_missing_dialogs.py — write the two dialog modules that
daw.py imports but which do not exist yet.

Creates:
    ui/import_stems_dialog.py
    ui/batch_dialog.py

Both are self-contained and depend only on ui.style and (for batch)
ui.dialogs, so they will not fail on an import cycle. Backs up existing
files if they happen to exist.
"""
import os
import shutil
import sys
import time


IMPORT_DIALOG = '''"""Import audio files into the current project.

Two modes:
  - Add as new track: each file becomes its own track.
  - Add to existing track: each file becomes an additional clip on a
    chosen track.

The dialog does not mutate the project. It exposes a plan
(`selected_files`, `import_mode`, `target_track_name`) that the caller
applies, so undo integration stays in one place (DAWWindow.import_stems).
"""
import os

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QListWidget, QListWidgetItem,
                             QFileDialog, QComboBox, QRadioButton,
                             QButtonGroup, QMessageBox)

from .style import COLORS


AUDIO_EXTS = ('.wav', '.flac', '.ogg', '.aiff', '.aif',
              '.mp3', '.m4a', '.aac', '.mp4', '.m4b', '.opus', '.wma')


class ImportStemsDialog(QDialog):
    def __init__(self, project, current_playhead=0.0, parent=None):
        super().__init__(parent)
        self.project = project
        self.playhead = current_playhead
        self.files = []
        self.setWindowTitle("Import Stems")
        self.setMinimumSize(620, 460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("Import Stems")
        title.setObjectName("header_title")
        layout.addWidget(title)

        hint = QLabel(
            "Add audio files to the current project. WAV, FLAC and OGG "
            "are read directly; MP3, M4A and other formats are converted "
            "with FFmpeg.")
        hint.setWordWrap(True)
        hint.setStyleSheet(
            "color: " + COLORS["text_dim"] + "; font-size: 12px;")
        layout.addWidget(hint)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(16)
        mode_row.addWidget(QLabel("Add as:"))
        self.mode_group = QButtonGroup(self)
        self.radio_new = QRadioButton("New track per file")
        self.radio_existing = QRadioButton("Clips on an existing track")
        self.radio_new.setChecked(True)
        self.mode_group.addButton(self.radio_new, 0)
        self.mode_group.addButton(self.radio_existing, 1)
        mode_row.addWidget(self.radio_new)
        mode_row.addWidget(self.radio_existing)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target track:"))
        self.target_combo = QComboBox()
        self._refresh_target_combo()
        target_row.addWidget(self.target_combo, 1)
        layout.addLayout(target_row)

        self.list = QListWidget()
        self.list.setStyleSheet(
            "QListWidget {{ background-color: {bg}; "
            "border: 1px solid {border}; border-radius: 6px; "
            "padding: 6px; font-size: 12px; }}".format(
                bg=COLORS["bg_alt"], border=COLORS["border"]))
        layout.addWidget(self.list, 1)

        file_btns = QHBoxLayout()
        btn_add = QPushButton("Add Files...")
        btn_add.clicked.connect(self.add_files)
        file_btns.addWidget(btn_add)
        btn_add_dir = QPushButton("Add Folder...")
        btn_add_dir.clicked.connect(self.add_folder)
        file_btns.addWidget(btn_add_dir)
        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(self.remove_selected)
        file_btns.addWidget(btn_remove)
        file_btns.addStretch()
        layout.addLayout(file_btns)

        pos_note = QLabel(
            "Clips will start at the current playhead "
            "({:.2f} s).".format(self.playhead))
        pos_note.setStyleSheet(
            "color: " + COLORS["text_dim"] + "; font-size: 12px;")
        layout.addWidget(pos_note)

        action_row = QHBoxLayout()
        action_row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        action_row.addWidget(cancel)
        self.btn_import = QPushButton("Import")
        self.btn_import.setObjectName("primary")
        self.btn_import.clicked.connect(self._on_import)
        action_row.addWidget(self.btn_import)
        layout.addLayout(action_row)

        self.radio_new.toggled.connect(self._update_mode)
        self.radio_existing.toggled.connect(self._update_mode)
        self._update_mode()

    def _update_mode(self):
        is_existing = self.radio_existing.isChecked()
        self.target_combo.setEnabled(
            is_existing and self.target_combo.count() > 0)

    def _refresh_target_combo(self):
        self.target_combo.clear()
        for t in self.project.tracks:
            self.target_combo.addItem(t.name.capitalize(), t.name)

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add audio files", os.path.expanduser("~"),
            "Audio (*.wav *.flac *.ogg *.aiff *.aif *.mp3 *.m4a *.aac "
            "*.mp4 *.m4b *.opus *.wma);;All files (*)")
        for p in paths:
            self._add(p)

    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add folder")
        if not d:
            return
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isfile(p) and p.lower().endswith(AUDIO_EXTS):
                self._add(p)

    def _add(self, path):
        if path not in self.files:
            self.files.append(path)
            self.list.addItem(QListWidgetItem(os.path.basename(path)))

    def remove_selected(self):
        for item in self.list.selectedItems():
            row = self.list.row(item)
            self.list.takeItem(row)
            del self.files[row]

    def _on_import(self):
        if not self.files:
            QMessageBox.information(
                self, "Nothing to import", "Add some audio files first.")
            return
        if self.radio_existing.isChecked() and self.target_combo.count() == 0:
            QMessageBox.warning(
                self, "No target track",
                "This project has no tracks yet. Choose "
                "\\u201cNew track per file\\u201d instead.")
            return
        self.accept()

    def selected_files(self):
        return list(self.files)

    def import_mode(self):
        return "existing_track" if self.radio_existing.isChecked() else "new_track"

    def target_track_name(self):
        return self.target_combo.currentData()
'''


BATCH_DIALOG = '''"""Batch separation: queue many audio files, run them one after another.

The worker calls core.separation.separate_file for each item. Uses the
same model selection dialog as single-file separation so the user picks
model / provider / chunked once for the whole batch.
"""
import os

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QListWidget, QListWidgetItem,
                             QProgressBar, QFileDialog)

from .style import COLORS


AUDIO_EXTS = ('.mp3', '.m4a', '.mp4', '.m4b', '.aac',
              '.wav', '.flac', '.ogg', '.opus', '.wma')


class _BatchWorker(QThread):
    item_started = pyqtSignal(int, str)
    item_progress = pyqtSignal(int, int, str)
    item_done = pyqtSignal(int, str)
    item_failed = pyqtSignal(int, str)
    all_done = pyqtSignal()

    def __init__(self, jobs, model, providers, chunked):
        super().__init__()
        self.jobs = jobs
        self.model = model
        self.providers = providers
        self.chunked = chunked
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        from core.separation import separate_file
        for i, (src, out_dir) in enumerate(self.jobs):
            if self._cancel:
                self.item_failed.emit(i, "Cancelled")
                continue
            self.item_started.emit(i, os.path.basename(src))
            try:
                separate_file(
                    src, out_dir, self.model,
                    providers=self.providers,
                    chunked=self.chunked,
                    progress_cb=lambda p, m, idx=i:
                        self.item_progress.emit(idx, p, m))
                self.item_done.emit(i, out_dir)
            except Exception as e:
                self.item_failed.emit(i, str(e))
        self.all_done.emit()


class BatchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Separation")
        self.setMinimumSize(760, 540)
        self.jobs = []
        self.worker = None
        self.output_root = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("Batch Separation")
        title.setObjectName("header_title")
        layout.addWidget(title)

        add_row = QHBoxLayout()
        self.btn_add_files = QPushButton("Add Files...")
        self.btn_add_files.clicked.connect(self.add_files)
        add_row.addWidget(self.btn_add_files)
        self.btn_add_folder = QPushButton("Add Folder...")
        self.btn_add_folder.clicked.connect(self.add_folder)
        add_row.addWidget(self.btn_add_folder)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_list)
        add_row.addWidget(self.btn_clear)
        add_row.addStretch()
        layout.addLayout(add_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output root:"))
        self.out_label = QLabel("(next to each source)")
        self.out_label.setStyleSheet(
            "color: " + COLORS["text_dim"] + ";")
        out_row.addWidget(self.out_label, 1)
        btn_out = QPushButton("Choose...")
        btn_out.clicked.connect(self.choose_output_root)
        out_row.addWidget(btn_out)
        layout.addLayout(out_row)

        self.list = QListWidget()
        self.list.setStyleSheet(
            "QListWidget {{ background-color: {bg}; "
            "border: 1px solid {border}; border-radius: 6px; "
            "padding: 6px; font-size: 12px; }}".format(
                bg=COLORS["bg_alt"], border=COLORS["border"]))
        layout.addWidget(self.list, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("")
        self.status.setStyleSheet(
            "color: " + COLORS["text_dim"] + "; font-size: 12px;")
        layout.addWidget(self.status)

        btns = QHBoxLayout()
        btns.addStretch()
        self.btn_start = QPushButton("Start Batch")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self.start)
        btns.addWidget(self.btn_start)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self.cancel)
        btns.addWidget(self.btn_cancel)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        btns.addWidget(self.btn_close)
        layout.addLayout(btns)

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add audio files", os.path.expanduser("~"),
            "Audio (*.mp3 *.m4a *.mp4 *.m4b *.aac *.wav *.flac *.ogg "
            "*.opus *.wma);;All files (*)")
        for p in paths:
            self._add(p)

    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add folder")
        if not d:
            return
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isfile(p) and p.lower().endswith(AUDIO_EXTS):
                self._add(p)

    def _add(self, path):
        base = os.path.splitext(os.path.basename(path))[0]
        if self.output_root:
            out_dir = os.path.join(self.output_root, base + "_stems")
        else:
            out_dir = os.path.splitext(path)[0] + "_stems"
        self.jobs.append((path, out_dir, "Pending"))
        self._refresh()

    def _refresh(self):
        self.list.clear()
        for path, out_dir, status in self.jobs:
            icon = {"Pending": "\\u2026", "Running": ">",
                    "Done": "OK", "Failed": "X"}.get(status, "?")
            self.list.addItem(QListWidgetItem(
                "{}  {}  \\u2192  {}".format(
                    icon, os.path.basename(path), out_dir)))

    def clear_list(self):
        if self.worker and self.worker.isRunning():
            return
        self.jobs.clear()
        self._refresh()

    def choose_output_root(self):
        d = QFileDialog.getExistingDirectory(self, "Output root")
        if d:
            self.output_root = d
            self.out_label.setText(d)
            self.out_label.setStyleSheet(
                "color: " + COLORS["text"] + ";")

    def start(self):
        if not self.jobs:
            self.status.setText("Add some files first.")
            return
        from core.resources import check_system_resources
        from ui.dialogs import ModelSelectionDialog
        resources = check_system_resources()
        dlg = ModelSelectionDialog(resources, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        model = dlg.selected_model()
        providers = ["cpu"] if dlg.force_cpu() else "auto"
        chunked = dlg.use_chunked()

        for i, (p, o, s) in enumerate(self.jobs):
            self.jobs[i] = (p, o, "Pending")
        self._refresh()

        self.progress.setVisible(True)
        self.progress.setRange(0, len(self.jobs))
        self.progress.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_cancel.setVisible(True)
        self.status.setText("Running batch...")

        jobs = [(p, o) for p, o, _ in self.jobs]
        self.worker = _BatchWorker(jobs, model, providers, chunked)
        self.worker.item_started.connect(self._on_started)
        self.worker.item_progress.connect(self._on_progress)
        self.worker.item_done.connect(self._on_done)
        self.worker.item_failed.connect(self._on_failed)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.status.setText("Cancelling after current item...")

    def _set_status(self, idx, status):
        p, o, _ = self.jobs[idx]
        self.jobs[idx] = (p, o, status)
        self._refresh()

    def _on_started(self, i, name):
        self._set_status(i, "Running")
        self.status.setText("Separating: " + name)

    def _on_progress(self, i, pct, msg):
        self.status.setText("{}: {}%".format(
            os.path.basename(self.jobs[i][0]), pct))

    def _on_done(self, i, out_dir):
        self._set_status(i, "Done")
        self.progress.setValue(self.progress.value() + 1)

    def _on_failed(self, i, err):
        self._set_status(i, "Failed")
        self.progress.setValue(self.progress.value() + 1)

    def _on_all_done(self):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setVisible(False)
        self.progress.setVisible(False)
        done = sum(1 for _, _, s in self.jobs if s == "Done")
        failed = sum(1 for _, _, s in self.jobs if s == "Failed")
        self.status.setText(
            "Batch complete: {} done, {} failed.".format(done, failed))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        event.accept()
'''


def backup(path):
    if not os.path.exists(path):
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst = path + ".bak." + stamp
    shutil.copy2(path, dst)
    print("  backup:", dst)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    backup(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("  wrote:", path, "({} bytes)".format(len(text)))


def main():
    if not os.path.isdir("ui") or not os.path.isdir("core"):
        print("ERROR: run from the project root.")
        sys.exit(1)

    print("Writing missing UI modules...")
    write(os.path.join("ui", "import_stems_dialog.py"), IMPORT_DIALOG)
    write(os.path.join("ui", "batch_dialog.py"), BATCH_DIALOG)

    print("")
    print("Verifying syntax...")
    for f in ("ui/import_stems_dialog.py", "ui/batch_dialog.py"):
        try:
            compile(open(f, "r", encoding="utf-8").read(), f, "exec")
            print("  {}: OK".format(f))
        except SyntaxError as e:
            print("  {}: SYNTAX ERROR at line {}".format(f, e.lineno))
            print("    ", e.msg)
            sys.exit(1)

    print("")
    print("Done. Both dialogs created.")
    print("Run: python apps/daw.py")


if __name__ == "__main__":
    main()
