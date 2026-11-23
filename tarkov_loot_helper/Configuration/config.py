"""User settings, persisted to settings.json in the project root."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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
    # Minimum icon-match score (0..1) for an item whose name couldn't be read.
    min_confidence: float = 0.70
    # Show low-confidence guesses in the overlay (dimmed, with a "?").
    show_uncertain: bool = True
    # Guesses worth less than this (RUB) are accepted at moderate confidence: mixing up
    # two cheap items (e.g. Glock 9x19 vs .45 mags) costs little. Valuable items stay strict.
    low_value_threshold: int = 10000
    # Item IDs that are never shown or exported. Default: WI-FI Camera (special slot).
    blacklist: list[str] = field(default_factory=lambda: ["5b4391a586f7745321235ab2"])
    # Highlight items worth at least this many roubles per slot.
    valuable_per_slot: int = 30000
    # Selected tarkov.dev game mode, and the options reported by /endpoints "gameModes".
    game_mode: str = "regular"
    available_game_modes: list[str] = field(default_factory=lambda: ["regular", "pve", "pvp-season"])
    language: str = "en"
    available_languages: list[str] = field(default_factory=lambda: ["en"])

    def cell_px(self, screen_height: int) -> float:
        return self.base_cell_px * (screen_height / 1080.0) * self.ui_scale

    @classmethod
    def load(cls) -> "Settings":
        if SETTINGS_FILE.exists():
            raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            known = {f.name for f in fields(cls)}
            if raw.get("min_confidence") == 0.55:  # old default was too permissive
                raw["min_confidence"] = 0.70
            return cls(**{k: v for k, v in raw.items() if k in known})
        return cls()

    def refresh_api_options(self, force: bool = False) -> bool:
        """Update game mode / language options from tarkov.dev /endpoints. Returns True if changed."""
        from tarkov_loot_helper.Application.TarkovDevAPIConnector import TarkovDevAPIConnector
        api = TarkovDevAPIConnector()
        modes, langs = api.game_modes(force), api.languages(force)
        changed = (modes and modes != self.available_game_modes) or (langs and langs != self.available_languages)
        if modes:
            self.available_game_modes = modes
            if self.game_mode not in modes:
                self.game_mode = modes[0]
        if langs:
            self.available_languages = langs
            if self.language not in langs:
                self.language = "en" if "en" in langs else langs[0]
        if changed:
            self.save()
        return bool(changed)

    def save(self) -> None:
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
