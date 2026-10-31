"""Loop markers on the time ruler."""
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from .style import COLORS
from .timeline import x_to_time, time_to_x


class LoopMarkers(QWidget):
    changed = pyqtSignal(float, float)
    clicked = pyqtSignal()

    def __init__(self, view, ruler, parent=None):
        super().__init__(parent)
        self.view = view
        self.ruler = ruler
        self.start = 0.0
        self.end = 0.0
        self.enabled = False
        self._drag = None
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setFixedHeight(ruler.height())
        view.horizontalScrollBar().valueChanged.connect(self.update)
        view.zoom_changed.connect(self.update)

    def set_loop(self, start, end, enabled):
        self.start, self.end, self.enabled = start, end, enabled
        self.update()

    def _x_from_time(self, t):
        return self.view.mapFromScene(QPointF(time_to_x(t), 0)).x()

    def _time_from_x(self, x):
        return max(0.0, x_to_time(
            self.view.mapToScene(int(x), 0).x()))

    def paintEvent(self, e):
        if self.end <= self.start:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        x0 = self._x_from_time(self.start)
        x1 = self._x_from_time(self.end)
        h = self.height()
        if self.enabled:
            tint = QColor(COLORS["accent"])
            tint.setAlpha(38)
            p.fillRect(QRectF(x0, 0, x1 - x0, h), tint)
        for which, x in (("start", x0), ("end", x1)):
            colour = (QColor(COLORS["accent"]) if self.enabled
                      else QColor("#5a5c64"))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(colour))
            if which == "start":
                pts = [QPointF(x, 0), QPointF(x + 8, 0),
                       QPointF(x, 8)]
            else:
                pts = [QPointF(x, 0), QPointF(x - 8, 0),
                       QPointF(x, 8)]
            p.drawPolygon(QPolygonF(pts))
            p.setPen(QPen(colour, 1))
            p.drawLine(QPointF(x, 0), QPointF(x, h))

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        x = e.position().x()
        xs = self._x_from_time(self.start)
        xe = self._x_from_time(self.end)
        if abs(x - xs) <= 8:
            self._drag = "start"
        elif abs(x - xe) <= 8:
            self._drag = "end"
        else:
            self._drag = None
            self.clicked.emit()
            return
        e.accept()

    def mouseMoveEvent(self, e):
        if not self._drag:
            return
        t = self._time_from_x(e.position().x())
        if self._drag == "start":
            self.start = min(t, self.end - 0.1)
        else:
            self.end = max(t, self.start + 0.1)
        self.update()
        self.changed.emit(self.start, self.end)
        e.accept()

    def mouseReleaseEvent(self, e):
        self._drag = None
        e.accept()
