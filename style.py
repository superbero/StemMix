# style.py
"""QSS stylesheet and color palette for the modern dark theme."""

# Color palette — a dark audio-app palette
COLORS = {
    "bg":        "#1e1e22",
    "bg_alt":    "#26262c",
    "bg_card":   "#2c2c34",
    "bg_hover":  "#35353e",
    "border":    "#3a3a44",
    "text":      "#e6e6ea",
    "text_dim":  "#9a9aa4",
    "accent":    "#4a9eff",
    "accent_hover": "#6cb0ff",
    "success":   "#5ed17a",
    "warning":   "#f0b655",
    "danger":    "#f06060",
    "track_colors": {
        "vocals": "#4a9eff",
        "drums":  "#f09055",
        "bass":   "#5ed17a",
        "other":  "#b07aff",
        "guitar": "#f06c9c",
        "piano":  "#f0d055",
    },
}


QSS = """
/* ---------- Global ---------- */
QWidget {
    background-color: %(bg)s;
    color: %(text)s;
    font-family: -apple-system, "Segoe UI", "Ubuntu", "Helvetica Neue", sans-serif;
    font-size: 14px;
}

QMainWindow {
    background-color: %(bg)s;
}

/* ---------- Labels ---------- */
QLabel {
    background: transparent;
    color: %(text)s;
}

QLabel#header_title {
    font-size: 20px;
    font-weight: 600;
    color: %(text)s;
    padding: 0px;
}

QLabel#header_subtitle {
    font-size: 12px;
    color: %(text_dim)s;
}

QLabel#status_bar {
    background-color: %(bg_alt)s;
    color: %(text_dim)s;
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 12px;
}

QLabel#progress_label {
    color: %(text_dim)s;
    font-size: 12px;
    padding: 4px 0;
}

QLabel#position_label {
    color: %(text_dim)s;
    font-size: 12px;
    font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
}

/* ---------- Buttons ---------- */
QPushButton {
    background-color: %(bg_card)s;
    color: %(text)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: %(bg_hover)s;
    border-color: %(accent)s;
}

QPushButton:pressed {
    background-color: %(bg_alt)s;
}

QPushButton:disabled {
    color: %(text_dim)s;
    background-color: %(bg_alt)s;
    border-color: %(border)s;
}

QPushButton#primary {
    background-color: %(accent)s;
    color: #ffffff;
    border: none;
    font-weight: 600;
    padding: 9px 20px;
}

QPushButton#primary:hover {
    background-color: %(accent_hover)s;
}

QPushButton#primary:disabled {
    background-color: %(border)s;
    color: %(text_dim)s;
}

/* ---------- Sliders ---------- */
QSlider::groove:horizontal {
    height: 4px;
    background: %(border)s;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: %(accent)s;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
    border: 2px solid %(accent)s;
}

QSlider::handle:horizontal:hover {
    background: %(accent_hover)s;
}

/* ---------- Progress bar ---------- */
QProgressBar {
    background-color: %(bg_alt)s;
    border: none;
    border-radius: 6px;
    height: 10px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background-color: %(accent)s;
    border-radius: 6px;
}

/* ---------- Group box (Tracks panel) ---------- */
QGroupBox {
    background-color: %(bg_alt)s;
    border: 1px solid %(border)s;
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 22px;
    font-weight: 600;
    font-size: 13px;
    color: %(text_dim)s;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 16px;
    padding: 0 6px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 11px;
    color: %(text_dim)s;
    background: transparent;
}

/* ---------- Checkboxes ---------- */
QCheckBox {
    background: transparent;
    spacing: 8px;
    font-size: 12px;
    color: %(text_dim)s;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid %(border)s;
    background: %(bg)s;
}

QCheckBox::indicator:hover {
    border-color: %(accent)s;
}

QCheckBox::indicator:checked {
    background: %(accent)s;
    border-color: %(accent)s;
    image: none;
}

QCheckBox:disabled {
    color: %(border)s;
}

/* ---------- Combo box ---------- */
QComboBox {
    background-color: %(bg_card)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
    padding: 7px 12px;
    font-size: 13px;
    min-width: 180px;
}

QComboBox:hover {
    border-color: %(accent)s;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: %(bg_card)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
    selection-background-color: %(accent)s;
    padding: 4px;
    outline: none;
}

/* ---------- Dialogs ---------- */
QDialog {
    background-color: %(bg)s;
}

/* ---------- Scroll area ---------- */
QScrollArea {
    background: transparent;
    border: none;
}

QScrollBar:vertical {
    background: %(bg_alt)s;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: %(border)s;
    border-radius: 5px;
    min-height: 24px;
}

QScrollBar::handle:vertical:hover {
    background: %(accent)s;
}

QScrollBar::add-line, QScrollBar::sub-line {
    height: 0;
}

/* ---------- Tooltip ---------- */
QToolTip {
    background-color: %(bg_card)s;
    color: %(text)s;
    border: 1px solid %(border)s;
    border-radius: 4px;
    padding: 6px 10px;
}
""" % COLORS