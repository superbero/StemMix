"""Import audio files into the current project.

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
                "\u201cNew track per file\u201d instead.")
            return
        self.accept()

    def selected_files(self):
        return list(self.files)

    def import_mode(self):
        return "existing_track" if self.radio_existing.isChecked() else "new_track"

    def target_track_name(self):
        return self.target_combo.currentData()
