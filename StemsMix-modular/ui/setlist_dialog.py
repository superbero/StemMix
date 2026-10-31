"""Persistent setlist editor for StemsMix project files."""
import os

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QMessageBox,
                             QPushButton, QVBoxLayout)

from core.project_io import FILE_EXT


def setlist_paths():
    raw = QSettings("StemsMix", "StemsMix").value(
        "setlist/projects", [], list) or []
    return [os.path.abspath(path) for path in raw
            if isinstance(path, str) and path]


class SetlistDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setlist")
        self.setMinimumSize(520, 400)
        self.selected_path = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Arrange StemsMix projects in performance order."))
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self._open_selected)
        layout.addWidget(self.list, 1)

        row = QHBoxLayout()
        for label, callback in (
                ("Add Projects…", self.add_projects),
                ("Remove", self.remove_selected),
                ("Move Up", lambda: self.move_selected(-1)),
                ("Move Down", lambda: self.move_selected(1))):
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)

        actions = QHBoxLayout()
        actions.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        actions.addWidget(close)
        open_button = QPushButton("Open Selected")
        open_button.setObjectName("primary")
        open_button.clicked.connect(self._open_selected)
        actions.addWidget(open_button)
        layout.addLayout(actions)
        self._populate(setlist_paths())

    def _populate(self, paths):
        self.list.clear()
        for path in paths:
            label = os.path.splitext(os.path.basename(path))[0]
            if not os.path.exists(path):
                label += " (missing)"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.list.addItem(item)

    def _paths(self):
        return [self.list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.list.count())]

    def _save(self):
        QSettings("StemsMix", "StemsMix").setValue(
            "setlist/projects", self._paths())

    def add_projects(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Projects to Setlist", os.path.expanduser("~"),
            "StemsMix Project (*{})".format(FILE_EXT))
        if not paths:
            return
        current = self._paths()
        for path in paths:
            path = os.path.abspath(path)
            if path not in current:
                current.append(path)
        self._populate(current)
        self._save()

    def remove_selected(self):
        row = self.list.currentRow()
        if row < 0:
            return
        self.list.takeItem(row)
        self._save()

    def move_selected(self, direction):
        row = self.list.currentRow()
        target = row + direction
        if row < 0 or not 0 <= target < self.list.count():
            return
        item = self.list.takeItem(row)
        self.list.insertItem(target, item)
        self.list.setCurrentRow(target)
        self._save()

    def _open_selected(self, *_args):
        item = self.list.currentItem()
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not os.path.isfile(path):
            QMessageBox.warning(
                self, "Project unavailable",
                "The setlist project cannot be found:\n\n" + path)
            return
        self._save()
        self.selected_path = path
        self.accept()
