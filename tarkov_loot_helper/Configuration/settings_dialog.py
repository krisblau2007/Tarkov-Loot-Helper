"""Settings window (UI scale, monitor, hotkeys, thresholds)."""
from __future__ import annotations

from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                             QLineEdit, QSpinBox)

from tarkov_loot_helper.Configuration.config import Settings


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tarkov Loot Helper - Settings")
        self.settings = settings
        form = QFormLayout(self)

        self.ui_scale = QDoubleSpinBox(minimum=0.5, maximum=2.0, singleStep=0.05, decimals=2)
        self.ui_scale.setValue(settings.ui_scale)
        form.addRow("In-game UI scale", self.ui_scale)

        self.base_cell = QSpinBox(minimum=30, maximum=120)
        self.base_cell.setValue(settings.base_cell_px)
        form.addRow("Cell size @1080p (px)", self.base_cell)

        self.monitor = QSpinBox(minimum=1, maximum=8)
        self.monitor.setValue(settings.monitor)
        form.addRow("Monitor", self.monitor)

        self.scan_key = QLineEdit(settings.scan_hotkey)
        form.addRow("Scan hotkey", self.scan_key)
        self.hide_key = QLineEdit(settings.hide_hotkey)
        form.addRow("Hide overlay hotkey", self.hide_key)

        self.seconds = QSpinBox(minimum=0, maximum=600)
        self.seconds.setValue(settings.overlay_seconds)
        self.seconds.setSpecialValueText("Until hidden")
        form.addRow("Overlay duration (s)", self.seconds)

        self.conf = QDoubleSpinBox(minimum=0.0, maximum=1.0, singleStep=0.05, decimals=2)
        self.conf.setValue(settings.min_confidence)
        form.addRow("Min match confidence", self.conf)

        self.valuable = QSpinBox(minimum=0, maximum=10_000_000, singleStep=5000)
        self.valuable.setValue(settings.valuable_per_slot)
        form.addRow("Valuable threshold (RUB/slot)", self.valuable)

        self.mode = QComboBox()
        self.mode.addItems(settings.available_game_modes)
        self.mode.setCurrentText(settings.game_mode)
        form.addRow("Game mode", self.mode)

        self.language = QComboBox()
        self.language.addItems(settings.available_languages)
        self.language.setCurrentText(settings.language)
        form.addRow("Item name language", self.language)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def accept(self) -> None:
        s = self.settings
        s.ui_scale = self.ui_scale.value()
        s.base_cell_px = self.base_cell.value()
        s.monitor = self.monitor.value()
        s.scan_hotkey = self.scan_key.text().strip() or "f8"
        s.hide_hotkey = self.hide_key.text().strip() or "f9"
        s.overlay_seconds = self.seconds.value()
        s.min_confidence = self.conf.value()
        s.valuable_per_slot = self.valuable.value()
        s.game_mode = self.mode.currentText()
        s.language = self.language.currentText()
        s.save()
        super().accept()
