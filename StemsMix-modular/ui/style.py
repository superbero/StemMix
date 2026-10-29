"""Palette and stylesheet. Graphite theme, one blue accent."""
COLORS = {
    "bg": "#16171a",
    "bg_alt": "#1d1e22",
    "bg_card": "#26272c",
    "bg_hover": "#303138",
    "border": "#2c2d33",
    "text": "#e4e5ea",
    "text_dim": "#8b8d97",
    "accent": "#4d9bff",
    "accent_hover": "#76b3ff",
    "success": "#4cd38a",
    "warning": "#f0b655",
    "danger": "#f06565",
    "lane_a": "#1b1c20",
    "lane_b": "#1f2024",
    "grid": "#2b2c32",
    "track_colors": {
        "vocals": "#4da3ff",
        "drums": "#ff9a52",
        "bass": "#4cd38a",
        "other": "#b383ff",
        "guitar": "#ff6f9f",
        "piano": "#ffd45c",
        "kick": "#ff6b5a",
        "snare": "#ffa14d",
        "toms": "#e2c653",
        "hh": "#5ed6cb",
        "ride": "#54b4e8",
        "crash": "#8f93ff",
    },
}

SCROLLBAR = 10  # px; the timeline and its header column rely on this

QSS = """
QWidget {
    background-color: %(bg)s; color: %(text)s;
    font-family: ".AppleSystemUIFont", "SF Pro Text", "Segoe UI",
                 "Ubuntu", "Helvetica Neue", sans-serif;
    font-size: 13px;
}
QMainWindow, QDialog { background-color: %(bg)s; }
QLabel { background: transparent; }
QLabel#header_title { font-size: 20px; font-weight: 600; }
QLabel#header_subtitle { font-size: 12px; color: %(text_dim)s; }
QLabel#status_bar {
    background-color: %(bg_alt)s; color: %(text_dim)s;
    padding: 8px 14px; border-radius: 6px; font-size: 12px;
}
QLabel#progress_label { color: %(text_dim)s; font-size: 12px; }
QLabel#position_label {
    color: %(text_dim)s; font-size: 12px;
    font-family: "SF Mono", "Menlo", "Consolas", monospace;
}

QPushButton {
    background-color: %(bg_card)s; color: %(text)s;
    border: 1px solid %(border)s; border-radius: 6px;
    padding: 6px 14px; font-size: 12px; font-weight: 500;
}
QPushButton:hover { background-color: %(bg_hover)s; }
QPushButton:pressed { background-color: %(bg_alt)s; }
QPushButton:disabled { color: #5a5c64; background-color: %(bg_alt)s; }
QPushButton#primary {
    background-color: %(accent)s; color: #ffffff;
    border: none; font-weight: 600; padding: 7px 16px;
}
QPushButton#primary:hover { background-color: %(accent_hover)s; }
QPushButton#primary:disabled { background-color: %(border)s; color: #5a5c64; }
QPushButton#transport {
    background: transparent; border: none; border-radius: 7px;
    padding: 0; min-width: 36px; max-width: 36px;
    min-height: 32px; max-height: 32px;
}
QPushButton#transport:hover { background-color: %(bg_hover)s; }
QPushButton#transport:pressed { background-color: %(bg_card)s; }
QPushButton#tiny {
    padding: 0; min-width: 26px; max-width: 26px;
    min-height: 22px; max-height: 22px; border-radius: 5px;
    font-size: 11px;
}
QPushButton#tinyWide {
    padding: 0 8px; min-height: 22px; max-height: 22px;
    border-radius: 5px; font-size: 11px;
}
QPushButton#msButton {
    background-color: %(bg_card)s; color: %(text_dim)s;
    border: 1px solid %(border)s; border-radius: 4px; padding: 0;
    min-width: 24px; max-width: 24px;
    min-height: 20px; max-height: 20px;
    font-size: 10px; font-weight: 700;
}
QPushButton#msButton:hover { background-color: %(bg_hover)s; }
QPushButton#msButton[kind="mute"]:checked {
    background-color: #e5534b; color: #ffffff; border-color: #e5534b;
}
QPushButton#msButton[kind="solo"]:checked {
    background-color: #f2c14e; color: #17181a; border-color: #f2c14e;
}

QFrame#transport { background-color: %(bg_alt)s;
                   border-bottom: 1px solid %(border)s; }
QLabel#brand { font-size: 15px; font-weight: 700; color: %(text)s; }
QLabel#timecode {
    font-family: "SF Mono", "Menlo", "Consolas", monospace;
    font-size: 24px; font-weight: 500; color: %(text)s;
}
QLabel#timecodeDim {
    font-family: "SF Mono", "Menlo", "Consolas", monospace;
    font-size: 13px; color: %(text_dim)s;
}
QFrame#timeBox { background-color: #101114; border: 1px solid %(border)s;
                 border-radius: 8px; }
QFrame#timeBox QLabel { background: transparent; }
QFrame#stepper { background-color: #101114;
                 border: 1px solid %(border)s; border-radius: 8px; }
QFrame#stepper QLabel { background: transparent; }
QFrame#stepper QPushButton#tiny { background: transparent; border: none;
                                  min-width: 22px; max-width: 22px; }
QFrame#stepper QPushButton#tiny:hover { background-color: %(bg_hover)s; }
QLabel#stepValue {
    font-family: "SF Mono", "Menlo", "Consolas", monospace;
    font-size: 14px; font-weight: 600; color: %(text)s;
}
QFrame#songbar { background-color: %(bg)s;
                 border-bottom: 1px solid %(border)s; }
QPushButton#clickButton { padding: 4px 12px 4px 8px; }
QPushButton#clickButton:checked { background-color: %(accent)s;
                                  color: #ffffff; border-color: %(accent)s; }
QLabel#beatDot { background-color: #2a2b31; border-radius: 6px; }
QLabel#beatDot[on="true"] { background-color: #ff5d5d; }
QLabel#beatDotAccent { background-color: #2a2b31; border-radius: 6px; }
QLabel#beatDotAccent[on="true"] { background-color: %(warning)s; }
QFrame#statusbar { background-color: %(bg_alt)s;
                   border-top: 1px solid %(border)s; }
QLabel#dawStatus { color: %(text_dim)s; font-size: 12px; }
QFrame#inspector { background-color: %(bg_alt)s;
                   border-top: 1px solid %(border)s;
                   border-bottom: 1px solid %(border)s; }
QLabel#sectionLabel { color: %(text_dim)s; font-size: 10px;
                      font-weight: 700; }
QLabel#clipName { font-weight: 600; }

QFrame#corner { background-color: %(bg_alt)s;
                border-right: 1px solid %(border)s;
                border-bottom: 1px solid %(border)s; }
QFrame#trackHeader { background-color: %(bg_alt)s;
                     border-bottom: 1px solid %(border)s;
                     border-right: 1px solid %(border)s; }
QLabel#trackName { font-weight: 600; font-size: 13px; }
QWidget#headerInner { background-color: %(lane_b)s; }
QFrame#mixer { background-color: %(bg_alt)s;
               border-top: 1px solid %(border)s; }
QWidget#mixerHolder { background: transparent; }
QFrame#strip { background-color: #222328; border: 1px solid %(border)s;
               border-radius: 8px; }
QLabel#dbReadout {
    font-family: "SF Mono", "Menlo", "Consolas", monospace;
    font-size: 10px; color: %(text_dim)s;
    background-color: #101114; border-radius: 4px; padding: 1px 0;
}

QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #3a3b43; border-radius: 4px;
                              min-height: 30px; margin: 1px; }
QScrollBar::handle:vertical:hover { background: #4a4b55; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: #3a3b43; border-radius: 4px;
                                min-width: 30px; margin: 1px; }
QScrollBar::handle:horizontal:hover { background: #4a4b55; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QSplitter::handle { background-color: %(border)s; }
QSplitter::handle:vertical { height: 3px; }
QToolTip { background-color: %(bg_card)s; color: %(text)s;
           border: 1px solid %(border)s; padding: 4px 6px; }

QSlider::groove:horizontal { height: 4px; background: %(border)s;
                             border-radius: 2px; }
QSlider::sub-page:horizontal { background: %(accent)s; border-radius: 2px; }
QSlider::handle:horizontal { background: #ffffff; width: 12px; height: 12px;
                             margin: -4px 0; border-radius: 6px; }
QDoubleSpinBox, QSpinBox {
    background-color: %(bg_card)s; border: 1px solid %(border)s;
    border-radius: 5px; padding: 3px 6px; min-width: 64px; font-size: 12px;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button { width: 0; border: none; }
QProgressBar { background-color: %(bg_card)s; border: none;
               border-radius: 3px; max-height: 6px; text-align: center;
               color: transparent; }
QProgressBar::chunk { background-color: %(accent)s; border-radius: 3px; }

QGroupBox {
    background-color: %(bg_alt)s; border: 1px solid %(border)s;
    border-radius: 10px; margin-top: 14px; padding-top: 22px;
    font-weight: 600; font-size: 13px; color: %(text_dim)s;
}
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
                   left: 16px; padding: 0 6px; font-size: 11px; }
QCheckBox { background: transparent; spacing: 8px; font-size: 12px;
            color: %(text_dim)s; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px;
                       border: 1px solid %(border)s; background: %(bg)s; }
QCheckBox::indicator:checked { background: %(accent)s;
                               border-color: %(accent)s; }
QComboBox { background-color: %(bg_card)s; border: 1px solid %(border)s;
            border-radius: 6px; padding: 7px 12px; font-size: 13px;
            min-width: 180px; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView { background-color: %(bg_card)s;
    border: 1px solid %(border)s; border-radius: 6px;
    selection-background-color: %(accent)s; padding: 4px; outline: none; }
""" % COLORS
