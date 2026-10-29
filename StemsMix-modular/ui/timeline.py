"""Arrange view: time ruler, lanes, clips, track headers."""
import math

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QFont, QPainter, QPainterPath,
                         QPen, QPolygonF, QTransform)
from PyQt6.QtWidgets import (QFrame, QGraphicsItem, QGraphicsPathItem,
                             QGraphicsScene, QGraphicsSimpleTextItem,
                             QGraphicsRectItem, QGraphicsView, QHBoxLayout,
                             QLabel, QPushButton, QScrollArea, QVBoxLayout,
                             QWidget)

from .style import COLORS, SCROLLBAR
from core.chords import transpose_chord

TRACK_HEIGHT = 72
CLIP_PAD = 5
CLIP_HEADER = 17
HEADER_WIDTH = 200
RULER_HEIGHT = 30
CHORD_HEIGHT = 28
PIXELS_PER_SECOND = 80          # scene units per second at zoom 1.0
MIN_TIMELINE_SECONDS = 120
EXTRA_TIMELINE_SECONDS = 30
MIN_ZOOM, MAX_ZOOM = 0.03, 16.0

_STEPS = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800]


def time_to_x(t):
    return t * PIXELS_PER_SECOND


def x_to_time(x):
    return x / PIXELS_PER_SECOND


def nice_step(px_per_sec, min_px=90):
    for s in _STEPS:
        if s * px_per_sec >= min_px:
            return s
    return _STEPS[-1]


def fmt_time(t, step):
    t = max(0.0, t)
    if step < 1:
        return "{}:{:04.1f}".format(int(t // 60), t % 60)
    return "{}:{:02d}".format(int(t) // 60, int(t) % 60)


class ClipItem(QGraphicsRectItem):
    def __init__(self, stem_name, audio_data, samplerate,
                 track_index, parent=None):
        duration = len(audio_data) / samplerate if samplerate else 0
        h = TRACK_HEIGHT - 2 * CLIP_PAD
        w = max(20.0, duration * PIXELS_PER_SECOND)
        super().__init__(0, 0, w, h, parent)
        self.stem_name = stem_name
        self.track_index = track_index
        self.duration = duration
        self.clip_ref = None
        self._base_w = w
        # No "move clip to another track" operation exists in the model,
        # so a clip is locked to its own lane (horizontal drag only).
        self._lane_y = track_index * TRACK_HEIGHT + CLIP_PAD
        self.base_color = QColor(COLORS["track_colors"].get(
            stem_name, COLORS["accent"]))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setY(self._lane_y)
        self.setX(0)

        wave = QGraphicsPathItem(self._build_waveform(audio_data, w, h), self)
        self._wave_item = wave
        line = QColor(self.base_color.lighter(165))
        fill = QColor(self.base_color.lighter(140))
        fill.setAlpha(150)
        pen = QPen(line, 1)
        pen.setCosmetic(True)
        wave.setPen(pen)
        wave.setBrush(QBrush(fill))
        wave.setZValue(1)
        wave.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        label = QGraphicsSimpleTextItem(stem_name.capitalize(), self)
        f = QFont()
        f.setBold(True)
        f.setPixelSize(11)
        label.setFont(f)
        label.setBrush(QBrush(QColor("#101114")))
        label.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        label.setPos(7, 1)
        label.setZValue(2)
        label.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    @staticmethod
    def _build_waveform(audio, w, h):
        """Min/max peak envelope below the clip's title bar."""
        path = QPainterPath()
        if audio is None or len(audio) == 0 or w < 4:
            return path
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        n_cols = int(max(2, min(w, len(mono) // 2)))
        edges = np.linspace(0, len(mono), n_cols + 1).astype(np.int64)[:-1]
        maxs = np.clip(np.maximum.reduceat(mono, edges), -1, 1)
        mins = np.clip(np.minimum.reduceat(mono, edges), -1, 1)
        area_top = CLIP_HEADER + 2
        center = (area_top + h) / 2.0
        scale = (h - area_top) / 2.0 - 2
        xs = np.linspace(0, w, n_cols)
        top = center - maxs * scale
        bot = center - mins * scale
        pts = [QPointF(float(x), float(y)) for x, y in zip(xs, top)]
        pts += [QPointF(float(x), float(y))
                for x, y in zip(xs[::-1], bot[::-1])]
        path.addPolygon(QPolygonF(pts))
        path.closeSubpath()
        return path

    def set_stretch(self, stretch):
        """Resize like Clip.end_time(): length = duration / stretch."""
        stretch = max(0.1, float(stretch))
        self.setRect(0, 0, max(20.0, self._base_w / stretch),
                     self.rect().height())
        self._wave_item.setTransform(QTransform.fromScale(1.0 / stretch, 1.0))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            return QPointF(max(0.0, value.x()), self._lane_y)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if self.clip_ref is not None:
                self.clip_ref.start_time = max(0.0, x_to_time(self.x()))
                sc = self.scene()
                if sc is not None and hasattr(sc, "clip_moved"):
                    sc.clip_moved.emit()
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        r = self.rect()
        sx = painter.worldTransform().m11() or 1.0
        rx, ry = 5.0 / sx, 5.0
        selected = self.isSelected()
        col = self.base_color
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        clip = QPainterPath()
        clip.addRoundedRect(r, rx, ry)
        painter.save()
        painter.setClipPath(clip)
        painter.fillRect(r, col.darker(260 if not selected else 215))
        painter.fillRect(QRectF(r.left(), r.top(), r.width(), CLIP_HEADER),
                         col if not selected else col.lighter(115))
        painter.restore()
        pen = QPen(QColor("#ffffff") if selected else col.darker(115),
                   1.6 if selected else 1.0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(r, rx, ry)


class PlayheadItem(QGraphicsItem):
    def __init__(self, height):
        super().__init__()
        self._h = height
        self.x_pos = 0.0
        self.setZValue(100)

    def boundingRect(self):
        return QRectF(self.x_pos - 8, 0, 16, self._h)

    def paint(self, painter, option, widget=None):
        pen = QPen(QColor("#ff5d5d"), 1.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(QPointF(self.x_pos, 0), QPointF(self.x_pos, self._h))

    def set_position(self, x):
        self.prepareGeometryChange()
        self.x_pos = x
        self.update()

    def set_height(self, h):
        self.prepareGeometryChange()
        self._h = h
        self.update()


class TimelineScene(QGraphicsScene):
    clip_moved = pyqtSignal()

    def __init__(self, num_tracks=6, parent=None):
        super().__init__(parent)
        self.num_tracks = num_tracks
        self._capacity_seconds = 0.0
        self.clips = []
        self._beat_x = None          # scene x of every beat (or None)
        self._down_x = None          # scene x of downbeats only
        self.playhead = PlayheadItem(num_tracks * TRACK_HEIGHT)
        self.addItem(self.playhead)
        self.set_extent(num_tracks, 0.0)

    def set_beats(self, times, downbeat_mask=None):
        """Show a beat grid (times in project seconds) or hide it (None)."""
        if times is None or len(times) < 2:
            self._beat_x = self._down_x = None
        else:
            x = np.asarray(times, dtype=float) * PIXELS_PER_SECOND
            self._beat_x = x
            self._down_x = x[np.asarray(downbeat_mask, dtype=bool)]
        self.update()

    def set_extent(self, num_tracks, seconds=0.0):
        """Size the scene for num_tracks lanes and `seconds` of audio."""
        n = max(1, int(num_tracks))
        self.num_tracks = n
        self._capacity_seconds = max(
            MIN_TIMELINE_SECONDS, seconds + EXTRA_TIMELINE_SECONDS)
        self.setSceneRect(0, 0, time_to_x(self._capacity_seconds),
                          n * TRACK_HEIGHT)
        self.playhead.set_height(n * TRACK_HEIGHT)
        self.update()

    def ensure_duration(self, seconds):
        if seconds + EXTRA_TIMELINE_SECONDS > self._capacity_seconds:
            self.set_extent(self.num_tracks, seconds)

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QColor(COLORS["lane_b"]))
        n, th = self.num_tracks, TRACK_HEIGHT
        first = max(0, int(rect.top() // th))
        last = min(n - 1, int(rect.bottom() // th))
        for i in range(first, last + 1):
            painter.fillRect(
                QRectF(rect.left(), i * th, rect.width(), th),
                QColor(COLORS["lane_a"] if i % 2 == 0 else COLORS["lane_b"]))
        pen = QPen(QColor(COLORS["grid"]), 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        pps = painter.worldTransform().m11() * PIXELS_PER_SECOND
        step = nice_step(pps)
        bottom = min(rect.bottom(), n * th)
        if self._beat_x is not None:
            self._draw_beats(painter, rect, bottom,
                             painter.worldTransform().m11())
        else:
            k = math.floor(rect.left() / PIXELS_PER_SECOND / step)
            while k * step * PIXELS_PER_SECOND <= rect.right():
                x = k * step * PIXELS_PER_SECOND
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, bottom))
                k += 1
        for i in range(first, last + 2):
            painter.drawLine(QPointF(rect.left(), i * th),
                             QPointF(rect.right(), i * th))

    def _draw_beats(self, painter, rect, bottom, sx):
        beat_px = float(np.median(np.diff(self._beat_x))) * sx
        pen_beat = QPen(QColor("#2e3037"), 1)
        pen_beat.setCosmetic(True)
        pen_down = QPen(QColor("#555965"), 1)
        pen_down.setCosmetic(True)
        for xs, pen, min_px in ((self._beat_x, pen_beat, 7.0),
                                (self._down_x, pen_down, 7.0)):
            if xs is None or len(xs) < 2:
                continue
            spacing = float(np.median(np.diff(xs))) * sx
            if spacing < min_px:
                continue
            i0 = int(np.searchsorted(xs, rect.left()))
            i1 = int(np.searchsorted(xs, rect.right()))
            painter.setPen(pen)
            for x in xs[i0:i1 + 1]:
                painter.drawLine(QPointF(float(x), rect.top()),
                                 QPointF(float(x), bottom))

    def add_clip(self, stem_name, audio_data, samplerate, track_index):
        c = ClipItem(stem_name, audio_data, samplerate, track_index)
        self.addItem(c)
        self.clips.append(c)
        return c

    def set_playhead(self, seconds):
        self.playhead.set_position(time_to_x(seconds))

    def clear_clips(self):
        for c in self.clips:
            self.removeItem(c)
        self.clips.clear()


class TimelineView(QGraphicsView):
    zoom_changed = pyqtSignal(float)
    chrome_changed = pyqtSignal(int)   # height of the horizontal scrollbar

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft
                          | Qt.AlignmentFlag.AlignTop)
        # Always visible, so its height is a constant the header column
        # can match (it is measured, not assumed, in resizeEvent).
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.zoom = 1.0

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.chrome_changed.emit(self.height() - self.viewport().height())

    def set_zoom(self, z, under_mouse=False):
        # Wheel zoom keeps the point under the cursor fixed; buttons zoom
        # around the centre of the view (an unanchored zoom with no mouse
        # over the view used to scroll it somewhere random).
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse if under_mouse
            else QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, z))
        self.setTransform(QTransform.fromScale(self.zoom, 1.0))
        self.zoom_changed.emit(self.zoom)

    def zoom_by(self, factor, under_mouse=False):
        self.set_zoom(self.zoom * factor, under_mouse)

    def fit_to(self, seconds):
        vw = max(200, self.viewport().width())
        self.set_zoom((vw - 30) / (max(seconds, 5.0) * PIXELS_PER_SECOND))
        self.horizontalScrollBar().setValue(0)
        self.verticalScrollBar().setValue(0)

    def wheelEvent(self, e):
        mods = e.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            self.zoom_by(1.15 if e.angleDelta().y() > 0 else 1 / 1.15,
                         under_mouse=True)
            e.accept()
        elif mods & Qt.KeyboardModifier.ShiftModifier:
            hb = self.horizontalScrollBar()
            hb.setValue(hb.value() - e.angleDelta().y())
            e.accept()
        else:
            super().wheelEvent(e)

    def follow(self, seconds):
        """Keep the playhead on screen while playing."""
        x = self.mapFromScene(QPointF(time_to_x(seconds), 0)).x()
        w = self.viewport().width()
        if x > w - 40 or x < 0:
            hb = self.horizontalScrollBar()
            hb.setValue(hb.value() + x - int(w * 0.2))


class TimeRuler(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view
        self.playhead = 0.0
        self.setFixedHeight(RULER_HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        view.horizontalScrollBar().valueChanged.connect(self.update)
        view.horizontalScrollBar().rangeChanged.connect(self.update)
        view.zoom_changed.connect(self.update)

    def set_playhead(self, t):
        self.playhead = t
        self.update()

    def _x(self, t):
        return self.view.mapFromScene(QPointF(time_to_x(t), 0)).x()

    def _t(self, x):
        return max(0.0, x_to_time(
            self.view.mapToScene(QPoint(int(x), 0)).x()))

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(COLORS["bg_alt"]))
        p.setPen(QColor(COLORS["border"]))
        p.drawLine(0, h - 1, w, h - 1)
        pps = PIXELS_PER_SECOND * self.view.zoom
        step = nice_step(pps)
        minor = step / 5.0
        i0 = int(math.floor(self._t(0) / minor))
        i1 = int(math.ceil(self._t(w) / minor))
        font = QFont()
        font.setPixelSize(10)
        p.setFont(font)
        for i in range(i0, i1 + 1):
            t = i * minor
            x = self._x(t)
            if i % 5 == 0:
                p.setPen(QColor("#5b5d67"))
                p.drawLine(x, h - 13, x, h - 1)
                p.setPen(QColor(COLORS["text_dim"]))
                p.drawText(x + 4, h - 15, fmt_time(t, step))
            else:
                p.setPen(QColor("#3d3e46"))
                p.drawLine(x, h - 6, x, h - 1)
        x = self._x(self.playhead)
        p.setPen(QPen(QColor("#ff5d5d"), 1.5))
        p.drawLine(x, 0, x, h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#ff5d5d"))
        p.drawPolygon(QPolygonF([QPointF(x - 6, 0), QPointF(x + 6, 0),
                                 QPointF(x, 9)]))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.seek_requested.emit(self._t(e.position().x()))

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self.seek_requested.emit(self._t(e.position().x()))


class ChordStrip(QWidget):
    """Chord lane under the ruler. Chords are stored in original-song
    time; they follow the project's tempo and are transposed with its key."""
    seek_requested = pyqtSignal(float)

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view
        self._chords = []
        self.tempo = 1.0
        self.key = 0.0
        self.playhead = 0.0
        self.setFixedHeight(CHORD_HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        view.horizontalScrollBar().valueChanged.connect(self.update)
        view.horizontalScrollBar().rangeChanged.connect(self.update)
        view.zoom_changed.connect(self.update)

    def set_chords(self, chords):
        self._chords = list(chords)
        self.update()

    def set_view_params(self, tempo, key):
        self.tempo, self.key = max(0.01, tempo), key
        self.update()

    def set_playhead(self, t):
        self.playhead = t
        self.update()

    def _x(self, t):
        return self.view.mapFromScene(QPointF(time_to_x(t), 0)).x()

    def _hit(self, x):
        for a, b, name in self._chords:
            if self._x(a / self.tempo) <= x < self._x(b / self.tempo):
                return a / self.tempo
        return None

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(COLORS["lane_b"]))
        p.setPen(QColor(COLORS["border"]))
        p.drawLine(0, h - 1, w, h - 1)
        font = QFont()
        font.setPixelSize(12)
        font.setBold(True)
        p.setFont(font)
        for a, b, name in self._chords:
            ta, tb = a / self.tempo, b / self.tempo
            x0, x1 = self._x(ta), self._x(tb)
            if x1 < 0 or x0 > w:
                continue
            rect = QRectF(x0 + 1, 3, max(2, x1 - x0 - 2), h - 7)
            now = ta <= self.playhead < tb
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(COLORS["accent"]) if now
                       else QColor(COLORS["bg_card"]))
            p.drawRoundedRect(rect, 4, 4)
            if rect.width() > 18:
                label = transpose_chord(name, self.key)
                p.setPen(QColor("#ffffff") if now else QColor(COLORS["text"]))
                text = p.fontMetrics().elidedText(
                    label, Qt.TextElideMode.ElideRight,
                    int(rect.width() - 10))
                p.drawText(rect.adjusted(6, 0, -4, 0),
                           Qt.AlignmentFlag.AlignVCenter
                           | Qt.AlignmentFlag.AlignLeft, text)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            t = self._hit(e.position().x())
            if t is not None:
                self.seek_requested.emit(t)


class TrackHeader(QFrame):
    mute_toggled = pyqtSignal(str, bool)
    solo_toggled = pyqtSignal(str, bool)

    def __init__(self, stem_name, color_hex, parent=None):
        super().__init__(parent)
        self.setObjectName("trackHeader")
        self.stem_name = stem_name
        self.setFixedHeight(TRACK_HEIGHT)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        strip = QFrame()
        strip.setFixedWidth(5)
        strip.setStyleSheet(
            "QFrame { background-color: %s; border: none; }" % color_hex)
        root.addWidget(strip)
        body = QVBoxLayout()
        body.setContentsMargins(12, 10, 10, 10)
        body.setSpacing(8)
        name = QLabel(stem_name.capitalize())
        name.setObjectName("trackName")
        body.addWidget(name)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.mute_btn = self._ms("M", "mute")
        self.solo_btn = self._ms("S", "solo")
        self.mute_btn.toggled.connect(
            lambda c: self.mute_toggled.emit(self.stem_name, c))
        self.solo_btn.toggled.connect(
            lambda c: self.solo_toggled.emit(self.stem_name, c))
        row.addWidget(self.mute_btn)
        row.addWidget(self.solo_btn)
        row.addStretch()
        body.addLayout(row)
        body.addStretch()
        root.addLayout(body, 1)

    @staticmethod
    def _ms(text, kind):
        b = QPushButton(text)
        b.setObjectName("msButton")
        b.setProperty("kind", kind)
        b.setCheckable(True)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return b

    def set_muted(self, v):
        self.mute_btn.blockSignals(True)
        self.mute_btn.setChecked(v)
        self.mute_btn.blockSignals(False)

    def set_soloed(self, v):
        self.solo_btn.blockSignals(True)
        self.solo_btn.setChecked(v)
        self.solo_btn.blockSignals(False)


class TrackHeaderColumn(QScrollArea):
    """Fixed-width header column that scrolls in lock-step with the view."""
    scrolled = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(HEADER_WIDTH)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        self.inner = QWidget()
        self.inner.setObjectName("headerInner")
        self._lay = QVBoxLayout(self.inner)
        # The view has a horizontal scrollbar at its bottom; set_bottom_pad
        # (driven by TimelineView.chrome_changed) matches its real height
        # so both scroll ranges are identical.
        self._lay.setContentsMargins(0, 0, 0, SCROLLBAR)
        self._lay.setSpacing(0)
        self._lay.addStretch()
        self.setWidget(self.inner)

    def set_bottom_pad(self, px):
        self._lay.setContentsMargins(0, 0, 0, max(0, int(px)))

    def clear(self):
        while self._lay.count() > 1:
            item = self._lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def add(self, header):
        self._lay.insertWidget(self._lay.count() - 1, header)

    def wheelEvent(self, e):
        bar = self.verticalScrollBar()
        bar.setValue(bar.value() - e.angleDelta().y())
        self.scrolled.emit(bar.value())
        e.accept()
