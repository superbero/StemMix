"""Custom-painted controls: Knob, Fader, LevelMeter, vector icons."""
import math

from PyQt6.QtCore import Qt, QPointF, QRectF, QSize, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QIcon, QLinearGradient, QPainter,
                         QPen, QPixmap, QPolygonF, QRadialGradient)
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel,
                             QPushButton, QToolTip, QWidget)

from .style import COLORS

DB_MIN, DB_MAX = -60.0, 6.0
_SPAN = DB_MAX - DB_MIN


def gain_to_pos(g):
    """Linear gain -> fader/meter position 0..1 (dB scale)."""
    if g <= 0.001:
        return 0.0
    db = 20.0 * math.log10(g)
    return max(0.0, min(1.0, (db - DB_MIN) / _SPAN))


def pos_to_gain(p):
    if p <= 0.001:
        return 0.0
    return 10 ** ((DB_MIN + p * _SPAN) / 20.0)


def fmt_db(g):
    if g <= 0.001:
        return "-inf"
    return "{:+.1f}".format(20.0 * math.log10(g))


class Knob(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, label, lo, hi, value=None, default=None,
                 fmt="{:.0f}", color=None, diameter=28, bipolar=False,
                 parent=None):
        super().__init__(parent)
        self.label = label
        self.lo, self.hi = float(lo), float(hi)
        self.default = float(lo if default is None else default)
        self._value = float(self.default if value is None else value)
        self._fmt = fmt if callable(fmt) else fmt.format
        self.color = QColor(color or COLORS["accent"])
        self.d = diameter
        self.bipolar = bipolar
        self._drag = None
        self.setFixedSize(diameter + 8, diameter + 14)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setToolTip("{}: {}".format(label, self._fmt(self._value)))

    def value(self):
        return self._value

    def set_value(self, v, emit=False):
        v = max(self.lo, min(self.hi, float(v)))
        if v == self._value:
            return
        self._value = v
        self.setToolTip("{}: {}".format(self.label, self._fmt(v)))
        self.update()
        if emit:
            self.valueChanged.emit(v)

    def _frac(self):
        return (self._value - self.lo) / (self.hi - self.lo)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        d = self.d
        cx = self.width() / 2.0
        cy = d / 2.0 + 1
        ring = QRectF(cx - d / 2 + 2, cy - d / 2 + 2, d - 4, d - 4)
        pen = QPen(QColor("#3a3b43"), 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(ring, 225 * 16, -270 * 16)
        f = self._frac()
        va = 225.0 - f * 270.0
        pen.setColor(self.color)
        p.setPen(pen)
        if self.bipolar:
            if abs(va - 90.0) > 0.6:
                p.drawArc(ring, 90 * 16, int((va - 90.0) * 16))
        elif f > 0.002:
            p.drawArc(ring, 225 * 16, int(-f * 270.0 * 16))
        body = ring.adjusted(5, 5, -5, -5)
        g = QRadialGradient(body.center().x(), body.top(), body.height())
        g.setColorAt(0.0, QColor("#6d6e79"))
        g.setColorAt(1.0, QColor("#3a3b43"))
        p.setPen(QPen(QColor("#121316"), 1))
        p.setBrush(QBrush(g))
        p.drawEllipse(body)
        c = body.center()
        r = body.width() / 2.0
        rad = math.radians(va)
        p1 = QPointF(c.x() + r * 0.30 * math.cos(rad),
                     c.y() - r * 0.30 * math.sin(rad))
        p2 = QPointF(c.x() + r * 0.92 * math.cos(rad),
                     c.y() - r * 0.92 * math.sin(rad))
        pen2 = QPen(QColor("#ffffff"), 2.0)
        pen2.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen2)
        p.drawLine(p1, p2)
        p.setPen(QColor(COLORS["text_dim"]))
        font = p.font()
        font.setPixelSize(9)
        font.setBold(True)
        p.setFont(font)
        p.drawText(QRectF(0, d + 1, self.width(), 12),
                   Qt.AlignmentFlag.AlignCenter, self.label)

    def _show_tip(self, e):
        QToolTip.showText(e.globalPosition().toPoint(),
                          "{}: {}".format(self.label,
                                          self._fmt(self._value)), self)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = (e.position().y(), self._value)
            self._show_tip(e)

    def mouseMoveEvent(self, e):
        if self._drag is None:
            return
        y0, v0 = self._drag
        fine = e.modifiers() & Qt.KeyboardModifier.ShiftModifier
        per_px = (self.hi - self.lo) / (600.0 if fine else 140.0)
        self.set_value(v0 + (y0 - e.position().y()) * per_px, emit=True)
        self._show_tip(e)

    def mouseReleaseEvent(self, e):
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        self.set_value(self.default, emit=True)

    def wheelEvent(self, e):
        step = (self.hi - self.lo) / 40.0
        self.set_value(self._value + (step if e.angleDelta().y() > 0
                                      else -step), emit=True)
        e.accept()


class Fader(QWidget):
    """Vertical dB fader with a scale. Emits linear gain (0..~2)."""
    valueChanged = pyqtSignal(float)
    PAD = 9
    TICKS = [(6, "+6"), (0, "0"), (-6, "6"), (-12, "12"),
             (-24, "24"), (-48, "48")]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pos = gain_to_pos(1.0)
        self.setFixedWidth(42)
        self.setMinimumHeight(110)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def gain(self):
        return pos_to_gain(self._pos)

    def set_gain(self, g, emit=False):
        pos = gain_to_pos(g)
        if abs(pos - self._pos) < 1e-6:
            return
        self._pos = pos
        self.update()
        if emit:
            self.valueChanged.emit(self.gain())

    def _set_pos(self, pos):
        self._pos = max(0.0, min(1.0, pos))
        self.update()
        self.valueChanged.emit(self.gain())

    def _from_y(self, y):
        top, bot = self.PAD, self.height() - self.PAD
        return (bot - y) / float(bot - top)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        top, bot = self.PAD, self.height() - self.PAD
        gx = 30
        font = p.font()
        font.setPixelSize(8)
        p.setFont(font)
        for db, txt in self.TICKS:
            y = bot - ((db - DB_MIN) / _SPAN) * (bot - top)
            p.setPen(QColor("#6d6f79"))
            p.drawText(QRectF(0, y - 6, 15, 12),
                       Qt.AlignmentFlag.AlignRight
                       | Qt.AlignmentFlag.AlignVCenter, txt)
            p.setPen(QPen(QColor("#4a4b53"), 1))
            p.drawLine(QPointF(17, y), QPointF(21, y))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#0d0e10"))
        p.drawRoundedRect(QRectF(gx - 2, top, 4, bot - top), 2, 2)
        cy = bot - self._pos * (bot - top)
        fill = QColor(COLORS["accent"])
        fill.setAlpha(170)
        p.setBrush(fill)
        p.drawRoundedRect(QRectF(gx - 2, cy, 4, bot - cy), 2, 2)
        cap = QRectF(gx - 11, cy - 7, 22, 14)
        g = QLinearGradient(0, cap.top(), 0, cap.bottom())
        g.setColorAt(0.0, QColor("#6a6b75"))
        g.setColorAt(1.0, QColor("#41424a"))
        p.setPen(QPen(QColor("#0f1012"), 1))
        p.setBrush(QBrush(g))
        p.drawRoundedRect(cap, 3, 3)
        p.setPen(QPen(QColor("#f4f5f8"), 1.5))
        p.drawLine(QPointF(cap.left() + 3, cy), QPointF(cap.right() - 3, cy))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._set_pos(self._from_y(e.position().y()))

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._set_pos(self._from_y(e.position().y()))
            QToolTip.showText(e.globalPosition().toPoint(),
                              fmt_db(self.gain()) + " dB", self)

    def mouseDoubleClickEvent(self, e):
        self._set_pos(gain_to_pos(1.0))

    def wheelEvent(self, e):
        self._set_pos(self._pos + (0.015 if e.angleDelta().y() > 0
                                   else -0.015))
        e.accept()


class LevelMeter(QWidget):
    """Stereo peak meter on the same dB scale and padding as Fader."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(14)
        self.setMinimumHeight(110)
        self._l = self._r = 0.0
        self._pl = self._pr = 0.0

    def set_levels(self, l, r):
        self._l = max(l, self._l * 0.80)
        self._r = max(r, self._r * 0.80)
        self._pl = max(l, self._pl * 0.985)
        self._pr = max(r, self._pr * 0.985)
        self.update()

    def reset(self):
        self._l = self._r = self._pl = self._pr = 0.0
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        top, bot = Fader.PAD, self.height() - Fader.PAD
        h = bot - top
        grad = QLinearGradient(0, bot, 0, top)
        grad.setColorAt(0.0, QColor("#2fc977"))
        grad.setColorAt(0.70, QColor("#3fd27f"))
        grad.setColorAt(0.80, QColor("#d6dd4a"))
        grad.setColorAt(0.895, QColor("#ffb02e"))
        grad.setColorAt(0.92, QColor("#ff4d4d"))
        grad.setColorAt(1.0, QColor("#ff4d4d"))
        for x, lvl, peak in ((1, self._l, self._pl), (8, self._r, self._pr)):
            p.fillRect(QRectF(x, top, 5, h), QColor("#0d0e10"))
            y = bot - gain_to_pos(lvl) * h
            if bot - y > 0.5:
                p.fillRect(QRectF(x, y, 5, bot - y), QBrush(grad))
            if peak > 0.002:
                py = bot - gain_to_pos(peak) * h
                p.fillRect(QRectF(x, py, 5, 1.5), QColor("#e9eaee"))


def make_icon(kind, color="#e4e5ea", size=18):
    """Vector transport/zoom icons, so they don't depend on fonts."""
    def draw(col):
        pm = QPixmap(size * 2, size * 2)
        pm.setDevicePixelRatio(2)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = float(size)
        c = QColor(col)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        if kind == "play":
            p.drawPolygon(QPolygonF([QPointF(s * .32, s * .2),
                                     QPointF(s * .8, s * .5),
                                     QPointF(s * .32, s * .8)]))
        elif kind == "pause":
            p.drawRoundedRect(QRectF(s * .27, s * .2, s * .17, s * .6), 1.5, 1.5)
            p.drawRoundedRect(QRectF(s * .56, s * .2, s * .17, s * .6), 1.5, 1.5)
        elif kind == "stop":
            p.drawRoundedRect(QRectF(s * .26, s * .26, s * .48, s * .48), 2, 2)
        elif kind == "rewind":
            p.drawRoundedRect(QRectF(s * .22, s * .22, s * .1, s * .56), 1, 1)
            p.drawPolygon(QPolygonF([QPointF(s * .78, s * .22),
                                     QPointF(s * .38, s * .5),
                                     QPointF(s * .78, s * .78)]))
        elif kind == "metronome":
            pen = QPen(c, 1.6)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolygon(QPolygonF([QPointF(s * .26, s * .86),
                                     QPointF(s * .74, s * .86),
                                     QPointF(s * .6, s * .14),
                                     QPointF(s * .4, s * .14)]))
            p.drawLine(QPointF(s * .5, s * .72), QPointF(s * .64, s * .3))
        elif kind in ("plus", "minus"):
            pen = QPen(c, 1.8)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(QPointF(s * .25, s * .5), QPointF(s * .75, s * .5))
            if kind == "plus":
                p.drawLine(QPointF(s * .5, s * .25), QPointF(s * .5, s * .75))
        p.end()
        return pm

    ic = QIcon()
    ic.addPixmap(draw(color), QIcon.Mode.Normal)
    ic.addPixmap(draw("#55575f"), QIcon.Mode.Disabled)
    return ic


class _DoubleClickLabel(QLabel):
    doubleClicked = pyqtSignal()

    def mouseDoubleClickEvent(self, e):
        self.doubleClicked.emit()


class Stepper(QFrame):
    """Compact  TITLE  [-] value [+]  control (tempo, key). Double-click
    the value to reset."""
    stepped = pyqtSignal(int)          # -1 or +1
    reset_requested = pyqtSignal()

    def __init__(self, title, tooltip="", parent=None):
        super().__init__(parent)
        self.setObjectName("stepper")
        self.setToolTip(tooltip)
        h = QHBoxLayout(self)
        h.setContentsMargins(10, 3, 6, 3)
        h.setSpacing(4)
        t = QLabel(title)
        t.setObjectName("sectionLabel")
        h.addWidget(t)
        self.minus = self._btn("minus", -1)
        self.plus = self._btn("plus", +1)
        self.value = _DoubleClickLabel("")
        self.value.setObjectName("stepValue")
        self.value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value.setFixedWidth(50)
        self.value.doubleClicked.connect(self.reset_requested.emit)
        h.addWidget(self.minus)
        h.addWidget(self.value)
        h.addWidget(self.plus)

    def _btn(self, kind, direction):
        b = QPushButton()
        b.setObjectName("tiny")
        b.setIcon(make_icon(kind, size=14))
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.clicked.connect(lambda: self.stepped.emit(direction))
        return b

    def set_text(self, text):
        self.value.setText(text)
