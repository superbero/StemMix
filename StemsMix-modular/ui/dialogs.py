"""Dialogs: prerequisites, model selection, export."""
import os

from PyQt6.QtCore import QThread, pyqtSignal, QSettings
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                             QLabel, QDialogButtonBox, QComboBox,
                             QCheckBox, QFormLayout, QScrollArea,
                             QWidget, QLineEdit, QPushButton,
                             QFileDialog, QPlainTextEdit, QProgressBar)

from .style import COLORS
from core.export import EXPORT_FORMATS, QUALITY_NOTES
from core.resources import (collect_install_plan, pip_install, brew_install,
                            run_prerequisite_checks, separation_backend_label)


class _InstallWorker(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, pip_pkgs, brew_pkgs):
        super().__init__()
        self.pip_pkgs = list(pip_pkgs)
        self.brew_pkgs = list(brew_pkgs)

    def run(self):
        logs = []
        if self.brew_pkgs:
            self.progress.emit(
                "Installing system packages: "
                + ", ".join(self.brew_pkgs))
            ok, log = brew_install(
                self.brew_pkgs,
                progress_cb=lambda m: self.progress.emit(m))
            logs.append(log)
            if not ok:
                self.failed.emit(
                    "brew install failed:\n" + (log[-2000:] if log else ""))
                return
        if self.pip_pkgs:
            self.progress.emit(
                "Installing Python packages: "
                + ", ".join(self.pip_pkgs))
            ok, log = pip_install(
                self.pip_pkgs,
                progress_cb=lambda m: self.progress.emit(m))
            logs.append(log)
            if not ok:
                self.failed.emit(
                    "pip install failed:\n" + (log[-2000:] if log else ""))
                return
        self.finished_ok.emit("\n".join(logs))


class PrerequisiteDialog(QDialog):
    STATUS_ICON = {"pass": "OK", "warn": "!", "fail": "X", "info": "i"}

    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = list(results)
        self.setWindowTitle("System Prerequisites")
        self.setMinimumWidth(680)
        self._install_worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        self.header = QLabel()
        self.header.setWordWrap(True)
        layout.addWidget(self.header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMaximumHeight(360)
        layout.addWidget(self.scroll)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        self.log.setPlaceholderText("Install log appears here…")
        self.log.setVisible(False)
        layout.addWidget(self.log)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.buttons = QDialogButtonBox()
        layout.addWidget(self.buttons)

        self._rebuild(self.results)

    def _set_header(self, results):
        blocking = any(r.blocking and r.status == "fail" for r in results)
        self.has_blocking_failure = blocking
        if blocking:
            text = ("Cannot start until FAIL items are fixed. "
                    "Use Install Missing to fetch Python packages.")
            bg = ("background-color: #3a1e22; color: #f8b6be; "
                  "border: 1px solid #5a2a30;")
        else:
            text = ("Some warnings detected. Install missing packages "
                    "for full speed and features, or continue anyway.")
            bg = ("background-color: #3a3020; color: #f0d090; "
                  "border: 1px solid #5a4a20;")
        self.header.setText(text)
        self.header.setStyleSheet(
            "QLabel { padding: 14px; border-radius: 8px; "
            "font-size: 13px; " + bg + " }")

    def _rebuild(self, results):
        self.results = list(results)
        self._set_header(results)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(6)
        for r in results:
            inner_layout.addWidget(self._make_row(r))
        inner_layout.addStretch()
        self.scroll.setWidget(inner)

        # Rebuild buttons
        for btn in list(self.buttons.buttons()):
            self.buttons.removeButton(btn)
            btn.deleteLater()

        pip_pkgs, brew_pkgs = collect_install_plan(results)
        self._pip_pkgs = pip_pkgs
        self._brew_pkgs = brew_pkgs

        if pip_pkgs or brew_pkgs:
            label_bits = []
            if pip_pkgs:
                label_bits.append("pip: " + ", ".join(pip_pkgs))
            if brew_pkgs:
                label_bits.append("brew: " + ", ".join(brew_pkgs))
            install = self.buttons.addButton(
                "Install Missing",
                QDialogButtonBox.ButtonRole.ActionRole)
            install.setObjectName("primary")
            install.setToolTip("Will install: " + " | ".join(label_bits))
            install.clicked.connect(self._start_install)

        if self.has_blocking_failure:
            self.buttons.addButton(
                "Quit", QDialogButtonBox.ButtonRole.RejectRole)
        else:
            cont = self.buttons.addButton(
                "Continue",
                QDialogButtonBox.ButtonRole.AcceptRole)
            cont.setObjectName("primary")
            cont.setDefault(True)
            self.buttons.addButton(
                "Quit", QDialogButtonBox.ButtonRole.RejectRole)

        try:
            self.buttons.accepted.disconnect()
        except TypeError:
            pass
        try:
            self.buttons.rejected.disconnect()
        except TypeError:
            pass
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def _make_row(self, r):
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(10)
        icon = QLabel(self.STATUS_ICON.get(r.status, "?"))
        icon.setFixedWidth(28)
        icon.setStyleSheet("font-weight: 700; font-size: 14px;")
        row.addWidget(icon)
        name = QLabel(r.name)
        name.setFixedWidth(110)
        row.addWidget(name)
        msg = QLabel(r.message)
        msg.setWordWrap(True)
        msg.setStyleSheet(
            "color: " + COLORS["text_dim"] + "; font-size: 12px;")
        row.addWidget(msg, 1)
        bg = {"pass": "#1e3a24", "warn": "#3a3020",
              "fail": "#3a1e22", "info": "#2a2a30"}.get(
                  r.status, COLORS["bg_card"])
        row_widget.setStyleSheet(
            "QWidget { background-color: " + bg +
            "; border-radius: 6px; }"
            "QLabel { background-color: transparent; }")
        return row_widget

    def _start_install(self):
        if self._install_worker and self._install_worker.isRunning():
            return
        self.log.setVisible(True)
        self.progress.setVisible(True)
        self.log.appendPlainText("Starting install…")
        for btn in self.buttons.buttons():
            btn.setEnabled(False)
        self._install_worker = _InstallWorker(
            self._pip_pkgs, self._brew_pkgs)
        self._install_worker.progress.connect(self.log.appendPlainText)
        self._install_worker.finished_ok.connect(self._install_done)
        self._install_worker.failed.connect(self._install_failed)
        self._install_worker.start()

    def _install_done(self, log):
        self.progress.setVisible(False)
        if log.strip():
            self.log.appendPlainText(log[-3000:])
        self.log.appendPlainText(
            "\nInstall finished. Re-checking… "
            "(restart the app if a package still shows missing)")
        # Fresh imports for newly installed modules
        results = run_prerequisite_checks()
        self._rebuild(results)
        for btn in self.buttons.buttons():
            btn.setEnabled(True)

    def _install_failed(self, message):
        self.progress.setVisible(False)
        self.log.appendPlainText(message)
        for btn in self.buttons.buttons():
            btn.setEnabled(True)


class OutputFolderDialog(QDialog):
    """Shown immediately after an audio file is chosen."""

    def __init__(self, input_file, parent=None):
        super().__init__(parent)
        self.settings = QSettings("StemsMix", "StemsMix")
        self.setWindowTitle("Choose Output Folder")
        self.setMinimumWidth(560)

        default_name = os.path.splitext(os.path.basename(input_file))[0]
        default_name = default_name + "_stems"
        last_parent = self.settings.value("last_output_dir", "", str)
        if last_parent and os.path.isdir(last_parent):
            initial = os.path.join(last_parent, default_name)
        else:
            initial = os.path.splitext(input_file)[0] + "_stems"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("Output Folder")
        title.setObjectName("header_title")
        layout.addWidget(title)

        info = QLabel(
            "Where should the separated stems for\n"
            "“{}” be saved?".format(os.path.basename(input_file)))
        info.setWordWrap(True)
        info.setStyleSheet(
            "color: " + COLORS["text_dim"] + "; font-size: 13px;")
        layout.addWidget(info)

        out_row = QHBoxLayout()
        out_row.setSpacing(8)
        self.output_edit = QLineEdit(initial)
        self.output_edit.setPlaceholderText(
            "Folder where stem WAV files will be written")
        self.output_edit.selectAll()
        out_row.addWidget(self.output_edit, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        out_row.addWidget(browse_btn)
        layout.addLayout(out_row)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet(
            "color: #f8b6be; font-size: 12px;")
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn:
            ok_btn.setObjectName("primary")
            ok_btn.setText("Continue")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self):
        start = self.output_edit.text().strip()
        if start and os.path.isdir(start):
            start_dir = start
        elif start:
            start_dir = os.path.dirname(start) or os.path.expanduser("~")
        else:
            start_dir = os.path.expanduser("~")
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose Output Folder", start_dir)
        if chosen:
            self.output_edit.setText(chosen)

    def output_dir(self):
        return self.output_edit.text().strip()

    def accept(self):
        path = self.output_dir()
        if not path:
            self.error_label.setText("Choose an output folder.")
            self.error_label.setVisible(True)
            return
        parent = os.path.dirname(path) or path
        self.settings.setValue("last_output_dir", parent)
        super().accept()


class ModelSelectionDialog(QDialog):
    def __init__(self, resources, parent=None):
        super().__init__(parent)
        self.resources = resources
        self.setWindowTitle("Choose Separation Model")
        self.setMinimumWidth(640)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("Separation Model")
        title.setObjectName("header_title")
        layout.addWidget(title)

        info = QLabel(
            "Fast (htdemucs) is the recommended default. High Quality "
            "(htdemucs_ft) runs four models and is roughly 4x slower "
            "for a small quality gain. 6-stem adds guitar and piano.")
        info.setWordWrap(True)
        info.setStyleSheet(
            "color: " + COLORS["text_dim"] + "; font-size: 13px;")
        layout.addWidget(info)

        sys_line = QLabel(
            "Detected: {:.1f} GB available RAM, {} CPU core(s).".format(
                resources["ram_gb"], resources["cores"]))
        sys_line.setStyleSheet(
            "color: " + COLORS["text_dim"] + "; font-size: 12px;")
        layout.addWidget(sys_line)

        self.combo = QComboBox()
        self.combo.addItem(
            "Fast 4-stem (htdemucs) - recommended", "htdemucs")
        self.combo.addItem(
            "High Quality 4-stem (htdemucs_ft) - ~4x slower",
            "htdemucs_ft")
        if resources["can_run_6_stem"]:
            self.combo.addItem(
                "Fine 6-stem (htdemucs_6s)", "htdemucs_6s")
        self.combo.currentIndexChanged.connect(self.on_model_changed)
        layout.addWidget(self.combo)

        # Opt-out of GPU — never auto-force CPU for any model.
        self.gpu_cb = QCheckBox(
            "Use GPU acceleration (Metal / MLX on Apple Silicon)")
        self.gpu_cb.setChecked(True)
        self.gpu_cb.toggled.connect(self._update_backend_hint)
        layout.addWidget(self.gpu_cb)

        self.chunked_cb = QCheckBox(
            "Use chunked separation (lower peak memory)")
        self.chunked_cb.setChecked(resources["ram_gb"] < 4)
        layout.addWidget(self.chunked_cb)

        self.backend_label = QLabel()
        self.backend_label.setWordWrap(True)
        self.backend_label.setStyleSheet(
            "color: " + COLORS["text_dim"] + "; font-size: 12px;")
        layout.addWidget(self.backend_label)

        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet(
            "QLabel { background-color: #3a3020; color: #f0d090; "
            "padding: 12px; border: 1px solid #5a4a20; "
            "border-radius: 6px; font-size: 12px; }")
        layout.addWidget(self.warning_label)

        if not resources["can_run_6_stem"]:
            blocked = QLabel(
                "6-stem mode is disabled: " +
                (resources["reason"] or ""))
            blocked.setWordWrap(True)
            blocked.setStyleSheet(
                "QLabel { background-color: #3a1e22; color: #f8b6be; "
                "padding: 12px; border: 1px solid #5a2a30; "
                "border-radius: 6px; font-size: 12px; }")
            layout.addWidget(blocked)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        ok_btn = buttons.button(
            QDialogButtonBox.StandardButton.Ok)
        if ok_btn:
            ok_btn.setObjectName("primary")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.on_model_changed()

    def on_model_changed(self):
        # Always keep GPU on when switching models — never auto-check CPU.
        self.gpu_cb.setChecked(True)
        model_name = self.combo.currentData()
        if model_name == "htdemucs_6s":
            self.warning_label.setText(
                "6-stem (guitar + piano). Keep GPU on for Metal/MLX speed. "
                "Uncheck GPU only if you run out of memory.")
        elif model_name == "htdemucs_ft":
            self.warning_label.setText(
                "High Quality runs 4 models (~4x slower than Fast). "
                "GPU stays on — do not confuse this with Force CPU.")
        else:
            self.warning_label.setText(
                "Fast 4-stem: best speed/quality balance on Apple Silicon.")
        self._update_backend_hint()

    def _update_backend_hint(self):
        force_cpu = not self.gpu_cb.isChecked()
        self.backend_label.setText(
            "Backend: " + separation_backend_label(force_cpu=force_cpu))

    def selected_model(self):
        return self.combo.currentData()

    def force_cpu(self):
        """True when the user opted out of GPU acceleration."""
        return not self.gpu_cb.isChecked()

    def use_chunked(self):
        return self.chunked_cb.isChecked()


class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Mix")
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("Export Mix")
        title.setObjectName("header_title")
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(10)
        self.format_combo = QComboBox()
        self.format_combo.addItems(EXPORT_FORMATS.keys())
        self.format_combo.currentTextChanged.connect(
            self.on_format_changed)
        form.addRow("Format:", self.format_combo)
        self.bitrate_combo = QComboBox()
        self.bitrate_combo.currentIndexChanged.connect(
            self.update_quality_label)
        form.addRow("Bitrate:", self.bitrate_combo)
        layout.addLayout(form)

        self.quality_label = QLabel()
        self.quality_label.setWordWrap(True)
        self.quality_label.setMinimumHeight(60)
        self.quality_label.setStyleSheet(
            "QLabel { background-color: " + COLORS["bg_card"] + "; "
            "color: " + COLORS["text_dim"] + "; padding: 12px; "
            "border: 1px solid " + COLORS["border"] + "; "
            "border-radius: 6px; font-size: 12px; }")
        layout.addWidget(self.quality_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        ok_btn = buttons.button(
            QDialogButtonBox.StandardButton.Ok)
        if ok_btn:
            ok_btn.setObjectName("primary")
            ok_btn.setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.on_format_changed(self.format_combo.currentText())

    def on_format_changed(self, fmt_key):
        self.bitrate_combo.blockSignals(True)
        self.bitrate_combo.clear()
        for label, value in EXPORT_FORMATS[fmt_key]["bitrates"]:
            self.bitrate_combo.addItem(label, value)
        self.bitrate_combo.blockSignals(False)
        self.update_quality_label()

    def update_quality_label(self):
        fmt = self.format_combo.currentText()
        br = self.bitrate_combo.currentText()
        self.quality_label.setText(
            QUALITY_NOTES.get((fmt, br), ""))

    def selected_format(self):
        return self.format_combo.currentText()

    def selected_bitrate(self):
        return self.bitrate_combo.currentData()
