"""
ARIA_hud.py  —  Always-on-top visual overlay for ARIA

A small, borderless, translucent corner window showing the assistant's
current state (idle / listening / thinking / speaking) plus the transcript
and reply text, updated as soon as each becomes available.

Qt must own the main thread on Windows, so ARIA_5.py runs its voice loop on
a background thread and posts state updates here through HUD_QUEUE; a QTimer
on the GUI thread drains it every TICK_MS and repaints. See push() below.

Controls
--------
    Ctrl+Alt+H            toggle the overlay on/off
    Left-click + drag      reposition
    Mouse wheel over it    resize (bounded)

Run standalone for a preview of all four states: python ARIA_hud.py
"""

import math
import queue
import sys
import time

import keyboard
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import QApplication, QWidget

HUD_QUEUE: "queue.Queue[dict]" = queue.Queue()

BASE_WIDTH, BASE_HEIGHT = 300, 170
MIN_SCALE, MAX_SCALE = 0.6, 2.0
TICK_MS = 33

STATE_COLORS = {
    "idle":      QColor(60, 200, 220),
    "listening": QColor(80, 220, 140),
    "thinking":  QColor(220, 180, 60),
    "speaking":  QColor(220, 90, 200),
}


class HUD(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        self.scale = 1.0
        self._resize_to_scale()
        self._move_to_corner()

        self.state = "idle"
        self.transcript = ""
        self.reply = ""
        self._rms = 0.0
        self._phase = 0.0
        self._drag_offset = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(TICK_MS)

        keyboard.add_hotkey("ctrl+alt+h", self._toggle_visible)

    # -- layout -------------------------------------------------------------
    def _resize_to_scale(self):
        self.setFixedSize(int(BASE_WIDTH * self.scale), int(BASE_HEIGHT * self.scale))

    def _move_to_corner(self):
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - self.width() - 24, screen.bottom() - self.height() - 24)

    def _toggle_visible(self):
        self.setVisible(not self.isVisible())

    # -- draining state updates ----------------------------------------------
    def _tick(self):
        self._phase += 0.12
        try:
            while True:
                msg = HUD_QUEUE.get_nowait()
                if "state" in msg:
                    self.state = msg["state"]
                if "transcript" in msg:
                    self.transcript = msg["transcript"]
                if "reply" in msg:
                    self.reply = msg["reply"]
                if "rms" in msg:
                    self._rms = msg["rms"]
        except queue.Empty:
            pass
        self.update()

    # -- mouse drag / wheel resize -------------------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event):
        self._drag_offset = None

    def wheelEvent(self, event):
        delta = 0.1 if event.angleDelta().y() > 0 else -0.1
        self.scale = max(MIN_SCALE, min(MAX_SCALE, self.scale + delta))
        self._resize_to_scale()

    # -- painting -------------------------------------------------------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = STATE_COLORS.get(self.state, STATE_COLORS["idle"])

        cx, cy = self.width() // 2, int(self.height() * 0.38)
        r = int(34 * self.scale)

        self._paint_ring(p, cx, cy, r, color)
        self._paint_text(p)

    def _paint_ring(self, p, cx, cy, r, color):
        if self.state == "listening":
            bars = 9
            amp = min(1.0, self._rms / 4000.0)
            p.setPen(QPen(color, 3))
            for i in range(bars):
                x = cx - r + i * (2 * r // (bars - 1))
                h = int(6 + amp * r * (0.4 + 0.6 * abs(math.sin(self._phase + i))))
                p.drawLine(x, cy - h, x, cy + h)

        elif self.state == "thinking":
            p.setPen(QPen(color, 4))
            span = 100
            start = int((self._phase * 60) % 360)
            p.drawArc(cx - r, cy - r, 2 * r, 2 * r, start * 16, span * 16)

        elif self.state == "speaking":
            pulse = (math.sin(self._phase * 2.2) + 1) / 2
            rr = int(r * (0.85 + 0.3 * pulse))
            p.setPen(QPen(color, 3))
            p.drawEllipse(QPoint(cx, cy), rr, rr)

        else:  # idle
            pulse = (math.sin(self._phase) + 1) / 2
            alpha = 90 + int(80 * pulse)
            p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), alpha), 3))
            p.drawEllipse(QPoint(cx, cy), r, r)

    def _paint_text(self, p):
        p.setPen(QColor(230, 230, 230, 220))
        p.setFont(QFont("Consolas", max(7, int(8 * self.scale))))
        top = int(self.height() * 0.62)
        w = self.width() - 20

        you = self._elide(self.transcript, 90)
        reply = self._elide(self.reply, 90)
        if you:
            p.drawText(10, top, w, 20, Qt.TextWordWrap, f"you: {you}")
        if reply:
            p.drawText(10, top + int(22 * self.scale), w, 40, Qt.TextWordWrap, f"aria: {reply}")

    @staticmethod
    def _elide(text, limit):
        text = (text or "").replace("\n", " ").strip()
        return text if len(text) <= limit else text[: limit - 1] + "…"


def push(**kwargs):
    """Called from the voice-loop thread to post a state update to the HUD."""
    HUD_QUEUE.put(kwargs)


_app = None
_hud = None


def start_hud():
    """Create the HUD window. Must be called on the main thread; the caller
    is then responsible for running app.exec()."""
    global _app, _hud
    _app = QApplication.instance() or QApplication(sys.argv)
    _hud = HUD()
    _hud.show()
    return _app


if __name__ == "__main__":
    app = start_hud()

    demo_states = ["idle", "listening", "thinking", "speaking"]

    def _demo():
        i = int(time.time() / 2) % len(demo_states)
        push(
            state=demo_states[i],
            transcript="what's the weather like today",
            reply="It's sunny and 21 degrees where you are.",
            rms=1800,
        )

    demo_timer = QTimer()
    demo_timer.timeout.connect(_demo)
    demo_timer.start(500)

    sys.exit(app.exec())
