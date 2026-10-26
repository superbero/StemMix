"""Timeline scene/view/clip widgets."""
from PyQt6.QtWidgets import (QGraphicsView, QGraphicsScene,
                             QGraphicsItem, QGraphicsRectItem,
                             QGraphicsLineItem, QGraphicsTextItem)
from PyQt6.QtGui import QBrush, QPen, QColor, QFont, QPainter
from PyQt6.QtCore import Qt, QPointF, QWheelEvent

from .style import COLORS


TRACK_HEIGHT = 90
PIXELS_PER_SECOND = 80


class ClipItem(QGraphicsRectItem):
    def __init__(self, stem_name, duration, track_index,
                 parent=None):
        h = TRACK_HEIGHT - 16
        w = max(20, duration * PIXELS_PER_SECOND)
        super().__init__(0, 0, w, h, parent)
        self.stem_name = stem_name
        self.track_index = track_index
        self.duration = duration

        color = QColor(COLORS["track_colors"].get(
            stem_name, COLORS["accent"]))
        self.setBrush(QBrush(color.darker(160)))
        self.setPen(QPen(color, 2))
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
            True)
        self.setY(track_index * TRACK_HEIGHT + 8)
        self.setX(8)

        label = QGraphicsTextItem(stem_name.capitalize(), self)
        label.setDefaultTextColor(QColor("#ffffff"))
        f = QFont()
        f.setBold(True)
        f.setPointSize(10)
        label.setFont(f)
        label.setPos(8, 4)

        dur_text = "{}:{:02d}".format(
            int(duration // 60), int(duration % 60))
        dl = QGraphicsTextItem(dur_text, self)
        dl.setDefaultTextColor(QColor("#cccccc"))
        f2 = QFont()
        f2.setPointSize(8)
        dl.setFont(f2)
        dl.setPos(8, h - 18)

    def itemChange(self, change, value):
        if change == \
                QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            new_pos = value
            lane = max(0, round(new_pos.y() / TRACK_HEIGHT))
            return QPointF(max(0, new_pos.x()),
                           lane * TRACK_HEIGHT + 8)
        if change == \
                QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.track_index = round(self.y() / TRACK_HEIGHT)
        return super().itemChange(change, value)


class TimelineScene(QGraphicsScene):
    def __init__(self, num_tracks=6, parent=None):
        super().__init__(parent)
        self.num_tracks = num_tracks
        self.setBackgroundBrush(QBrush(QColor(COLORS["bg"])))
        self.setSceneRect(
            0, 0, 10000, num_tracks * TRACK_HEIGHT + 20)

        for i in range(num_tracks):
            rect = QGraphicsRectItem(
                0, i * TRACK_HEIGHT, 10000, TRACK_HEIGHT)
            rect.setBrush(QBrush(QColor(
                COLORS["bg_alt"] if i % 2 == 0 else COLORS["bg"])))
            rect.setPen(QPen(Qt.PenStyle.NoPen))
            rect.setZValue(-10)
            self.addItem(rect)

        for i in range(num_tracks + 1):
            line = QGraphicsLineItem(
                0, i * TRACK_HEIGHT, 10000, i * TRACK_HEIGHT)
            line.setPen(QPen(QColor(COLORS["border"]), 1))
            line.setZValue(-5)
            self.addItem(line)

        self.playhead = QGraphicsLineItem(
            8, 0, 8, num_tracks * TRACK_HEIGHT)
        self.playhead.setPen(QPen(QColor("#ff4444"), 2))
        self.playhead.setZValue(100)
        self.addItem(self.playhead)

        self.clips = []

    def add_clip(self, stem_name, duration, track_index):
        c = ClipItem(stem_name, duration, track_index)
        self.addItem(c)
        self.clips.append(c)
        return c

    def set_playhead(self, seconds):
        self.playhead.setX(8 + seconds * PIXELS_PER_SECOND)

    def clear_clips(self):
        for c in self.clips:
            self.removeItem(c)
        self.clips.clear()


class TimelineView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(
            QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setStyleSheet(
            "QGraphicsView { background-color: " + COLORS["bg"] +
            "; border: 1px solid " + COLORS["border"] +
            "; border-radius: 8px; }")

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, 1.0)
            event.accept()
        else:
            super().wheelEvent(event)
