"""Dialogs: prerequisites, model selection, export."""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                             QLabel, QDialogButtonBox, QComboBox,
                             QCheckBox, QFormLayout, QScrollArea,
                             QWidget)

from .style import COLORS
from core.export import EXPORT_FORMATS, QUALITY_NOTES


class PrerequisiteDialog(QDialog):
    STATUS_ICON = {"pass": "OK", "warn": "!", "fail": "X", "info": "i"}

    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results
        self.setWindowTitle("System Prerequisites")
        self.setMinimumWidth(640)
        self.has_blocking_failure = any(
            r.blocking and r.status == "fail" for r in results)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        if self.has_blocking_failure:
            header_text = ("Cannot start. Fix the items marked FAIL.")
            bg_style = ("background-color: #3a1e22; "
                        "color: #f8b6be; border: 1px solid #5a2a30;")
        else:
            header_text = ("Some warnings detected. The app should work, "
                           "but some features may be limited.")
            bg_style = ("background-color: #3a3020; "
                        "color: #f0d090; border: 1px solid #5a4a20;")

        header = QLabel(header_text)
        header.setWordWrap(True)
        header.setStyleSheet(
            "QLabel { padding: 14px; border-radius: 8px; "
            "font-size: 13px; " + bg_style + " }")
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(360)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(6)
        for r in results:
            inner_layout.addWidget(self._make_row(r))
        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox()
        if self.has_blocking_failure:
            buttons.addButton(
                "Quit", QDialogButtonBox.ButtonRole.RejectRole)
        else:
            cont = buttons.addButton(
                "Continue anyway",
                QDialogButtonBox.ButtonRole.AcceptRole)
            cont.setObjectName("primary")
            cont.setDefault(True)
            buttons.addButton(
                "Quit", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

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


class ModelSelectionDialog(QDialog):
    def __init__(self, resources, parent=None):
        super().__init__(parent)
        self.resources = resources
        self.setWindowTitle("Choose Separation Model")
        self.setMinimumWidth(620)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("Separation Model")
        title.setObjectName("header_title")
        layout.addWidget(title)

        info = QLabel(
            "High Quality gives the best results. Fast is roughly 30% "
            "faster with slightly lower quality. 6-stem adds guitar "
            "and piano at the cost of higher memory use.")
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
            "High Quality 4-stem (htdemucs_ft)", "htdemucs_ft")
        self.combo.addItem(
            "Fast 4-stem (htdemucs)", "htdemucs")
        if resources["can_run_6_stem"]:
            self.combo.addItem(
                "Fine 6-stem (htdemucs_6s)", "htdemucs_6s")
        self.combo.currentIndexChanged.connect(self.on_model_changed)
        layout.addWidget(self.combo)

        self.force_cpu_cb = QCheckBox("Force CPU")
        layout.addWidget(self.force_cpu_cb)

        self.chunked_cb = QCheckBox(
            "Use chunked separation (lower peak memory)")
        self.chunked_cb.setChecked(resources["ram_gb"] < 8)
        layout.addWidget(self.chunked_cb)

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
        model_name = self.combo.currentData()
        if model_name == "htdemucs_6s":
            self.force_cpu_cb.setChecked(True)
            self.warning_label.setText(
                "6-stem is RAM-heavy. Force CPU is pre-selected.")
        elif model_name == "htdemucs":
            self.force_cpu_cb.setChecked(False)
            self.warning_label.setText(
                "Fast 4-stem: single file, ~30% faster.")
        else:
            self.force_cpu_cb.setChecked(False)
            self.warning_label.setText(
                "High Quality 4-stem: best separation quality.")

    def selected_model(self):
        return self.combo.currentData()

    def force_cpu(self):
        return self.force_cpu_cb.isChecked()

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
        self.bitrate_combo.currentIndexChanged.connect(
            self.update_quality_label)
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
