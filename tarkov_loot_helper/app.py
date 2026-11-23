"""System-tray application: global hotkey -> scan -> overlay."""
from __future__ import annotations

import sys
import threading
import traceback
from datetime import datetime

import cv2
import keyboard
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QColor, QIcon, QPixmap
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from . import api, export
from .capture import grab
from .config import DEBUG_DIR, Settings
from .identify import Identifier
from .overlay import Overlay
from .scanner import render_debug, scan_image
from .settings_dialog import SettingsDialog

UI_SCALE_PRESETS = [0.75, 0.85, 1.0, 1.1, 1.25, 1.5]


class Bridge(QObject):
    scan_requested = pyqtSignal()
    hide_requested = pyqtSignal()
    scan_done = pyqtSignal(object, object)
    message = pyqtSignal(str)


class App:
    def __init__(self):
        self.qt = QApplication(sys.argv)
        self.qt.setQuitOnLastWindowClosed(False)
        self.settings = Settings.load()
        self.identifier: Identifier | None = None
        self.last_result = None
        self.scanning = False
        self.save_debug = False

        self.bridge = Bridge()
        self.bridge.scan_requested.connect(self.start_scan)
        self.bridge.hide_requested.connect(lambda: self.overlay.hide())
        self.bridge.scan_done.connect(self.on_scan_done)
        self.bridge.message.connect(lambda m: self.tray.showMessage("Tarkov Loot Helper", m))

        self.overlay = Overlay(self.settings)
        self._build_tray()
        self._bind_hotkeys()
        threading.Thread(target=self._load_data, daemon=True).start()

    # ---- setup -------------------------------------------------------------
    def _build_tray(self):
        pix = QPixmap(32, 32)
        pix.fill(QColor(200, 170, 60))
        self.tray = QSystemTrayIcon(QIcon(pix))
        menu = QMenu()
        menu.addAction("Scan now", self.start_scan)
        menu.addAction("Export last scan", self.export_last)

        scale_menu = menu.addMenu("UI scale")
        self.scale_group = QActionGroup(scale_menu)
        for v in UI_SCALE_PRESETS:
            act = QAction(f"{v:.2f}", scale_menu, checkable=True)
            act.setChecked(abs(v - self.settings.ui_scale) < 1e-3)
            act.triggered.connect(lambda _=False, v=v: self.set_ui_scale(v))
            self.scale_group.addAction(act)
            scale_menu.addAction(act)

        dbg = QAction("Save debug images", menu, checkable=True)
        dbg.toggled.connect(lambda on: setattr(self, "save_debug", on))
        menu.addAction(dbg)
        menu.addAction("Refresh prices", lambda: threading.Thread(
            target=self._load_data, args=(True,), daemon=True).start())
        menu.addAction("Settings...", self.open_settings)
        menu.addSeparator()
        menu.addAction("Quit", self.qt.quit)
        self.tray.setContextMenu(menu)
        self.tray.setToolTip("Tarkov Loot Helper")
        self.tray.show()

    def _bind_hotkeys(self):
        keyboard.unhook_all_hotkeys()
        keyboard.add_hotkey(self.settings.scan_hotkey, self.bridge.scan_requested.emit)
        keyboard.add_hotkey(self.settings.hide_hotkey, self.bridge.hide_requested.emit)

    def _load_data(self, force: bool = False):
        try:
            self.bridge.message.emit("Loading item data from tarkov.dev...")
            items = api.load_items(self.settings.game_mode, force=force)
            self.identifier = Identifier(items)
            self.bridge.message.emit(
                f"Ready: {len(items)} items. Press {self.settings.scan_hotkey.upper()} with inventory open.")
        except Exception as e:  # noqa: BLE001
            self.bridge.message.emit(f"Failed to load item data: {e}")

    # ---- actions -----------------------------------------------------------
    def set_ui_scale(self, v: float):
        self.settings.ui_scale = v
        self.settings.save()
        self.bridge.message.emit(f"UI scale set to {v:.2f}")

    def open_settings(self):
        old_mode = self.settings.game_mode
        if SettingsDialog(self.settings).exec():
            self._bind_hotkeys()
            for act in self.scale_group.actions():
                act.setChecked(abs(float(act.text()) - self.settings.ui_scale) < 1e-3)
            if self.settings.game_mode != old_mode:
                threading.Thread(target=self._load_data, daemon=True).start()

    def start_scan(self):
        if self.scanning:
            return
        if self.identifier is None:
            self.bridge.message.emit("Item data is still loading, try again in a moment.")
            return
        self.scanning = True
        self.overlay.hide()  # don't capture our own overlay
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        try:
            img, mon = grab(self.settings.monitor)
            result = scan_image(img, self.settings, self.identifier)
            if self.save_debug:
                DEBUG_DIR.mkdir(exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(DEBUG_DIR / f"{stamp}_raw.png"), img)
                cv2.imwrite(str(DEBUG_DIR / f"{stamp}_annotated.png"), render_debug(result))
            self.bridge.scan_done.emit(result, mon)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self.bridge.message.emit("Scan failed - see console for details.")
            self.bridge.scan_done.emit(None, None)

    def on_scan_done(self, result, mon):
        self.scanning = False
        if result is None:
            return
        self.last_result = result
        if not result.grids:
            self.bridge.message.emit("No inventory grid found. Check the UI scale setting.")
            return
        self.overlay.show_result(result, mon)
        total = sum(d.item.best_price for d in result.identified)
        print(f"Scan: {len(result.grids)} grids, {len(result.detections)} items, "
              f"{len(result.identified)} identified, ~{total:,} RUB")

    def export_last(self):
        if not self.last_result:
            self.bridge.message.emit("Nothing scanned yet.")
            return
        csv_path, _ = export.export(self.last_result)
        self.bridge.message.emit(f"Exported to {csv_path}")

    def run(self) -> int:
        return self.qt.exec()
