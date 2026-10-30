"""Batch separation: queue many files, run them one after another."""
import os

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QListWidget, QListWidgetItem,
                             QProgressBar, QFileDialog, QMessageBox)

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
        self.jobs = jobs   # list of (input_path, output_dir)
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
        self.btn_add_files = QPushButton("Add Files…")
        self.btn_add_files.clicked.connect(self.add_files)
        add_row.addWidget(self.btn_add_files)
        self.btn_add_folder = QPushButton("Add Folder…")
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
        self.out_label.setStyleSheet(f"color: {COLORS['text_dim']};")
        out_row.addWidget(self.out_label, 1)
        btn_out = QPushButton("Choose…")
        btn_out.clicked.connect(self.choose_output_root)
        out_row.addWidget(btn_out)
        layout.addLayout(out_row)

        self.list = QListWidget()
        self.list.setStyleSheet(
            f"QListWidget {{ background-color: {COLORS['bg_alt']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 6px; "
            f"padding: 6px; font-size: 12px; }}")
        layout.addWidget(self.list, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
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
            icon = {"Pending": "…", "Running": ">",
                    "Done": "OK", "Failed": "X"}.get(status, "?")
            self.list.addItem(QListWidgetItem(
                "{}  {}  →  {}".format(
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
            self.out_label.setStyleSheet(f"color: {COLORS['text']};")

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
        self.status.setText("Running batch…")

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
            self.status.setText("Cancelling after current item…")

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
