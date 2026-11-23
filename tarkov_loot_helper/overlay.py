"""Transparent, click-through, always-on-top overlay that labels items."""
from __future__ import annotations

from PyQt6.QtCore import QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from .config import Settings
from .scanner import ScanResult


def fmt_rub(v: int) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1000:
        return f"{v // 1000}k"
    return str(v)


class Overlay(QWidget):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.result: ScanResult | None = None
        self.scale = 1.0  # physical px -> Qt logical px
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.timer = QTimer(self, singleShot=True, timeout=self.hide)

    def show_result(self, result: ScanResult, monitor: dict) -> None:
        self.result = result
        screen = self.screen()
        self.scale = 1.0 / (screen.devicePixelRatio() if screen else 1.0)
        self.setGeometry(int(monitor["left"] * self.scale), int(monitor["top"] * self.scale),
                         int(monitor["width"] * self.scale), int(monitor["height"] * self.scale))
        self.show()
        self.raise_()
        self.update()
        if self.settings.overlay_seconds > 0:
            self.timer.start(self.settings.overlay_seconds * 1000)

    def paintEvent(self, _event) -> None:
        if not self.result:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        p.setFont(font)
        s = self.scale
        for d in self.result.identified:
            it = d.item
            x, y, w, h = d.region.rect
            rect = QRect(int(x * s), int(y * s), int(w * s), int(h * s))
            if it.quest_needs:
                color = QColor(255, 200, 0)
            elif it.hideout_needs:
                color = QColor(80, 200, 255)
            elif it.price_per_slot >= self.settings.valuable_per_slot:
                color = QColor(60, 230, 90)
            else:
                color = QColor(200, 200, 200, 160)
            p.setPen(QPen(color, 2))
            p.drawRect(rect.adjusted(1, 1, -1, -1))

            tags = ("Q " if it.quest_needs else "") + ("H " if it.hideout_needs else "")
            label = f"{tags}{fmt_rub(it.price_per_slot)}/slot"
            band = QRect(rect.left() + 2, rect.bottom() - 15, rect.width() - 4, 14)
            p.fillRect(band, QColor(0, 0, 0, 190))
            p.setPen(color)
            p.drawText(band, Qt.AlignmentFlag.AlignCenter, label)
        p.end()
