"""User settings, persisted to settings.json in the project root."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ICON_DIR = DATA_DIR / "icons"
EXPORT_DIR = ROOT / "exports"
DEBUG_DIR = ROOT / "debug"
SETTINGS_FILE = ROOT / "settings.json"


@dataclass
class Settings:
    # Tarkov's in-game UI scale (Settings > Graphics/Game). 1.0 = default.
    ui_scale: float = 1.0
    # Size in pixels of one inventory cell at 1080p and UI scale 1.0.
    base_cell_px: int = 63
    # Monitor to capture (1 = primary in mss numbering).
    monitor: int = 1
    scan_hotkey: str = "f8"
    hide_hotkey: str = "f9"
    overlay_seconds: int = 15
    # Minimum match score (0..1) for an identification to be shown.
    min_confidence: float = 0.55
    # Highlight items worth at least this many roubles per slot.
    valuable_per_slot: int = 30000
    game_mode: str = "regular"  # "regular" or "pve"

    def cell_px(self, screen_height: int) -> float:
        return self.base_cell_px * (screen_height / 1080.0) * self.ui_scale

    @classmethod
    def load(cls) -> "Settings":
        if SETTINGS_FILE.exists():
            raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            known = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in raw.items() if k in known})
        return cls()

    def save(self) -> None:
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
