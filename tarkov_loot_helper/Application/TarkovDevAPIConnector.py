"""All communication with the public tarkov.dev JSON API (https://json.tarkov.dev),
with disk caching.

  /endpoints                   -> list of endpoints, gameModes and languages
  /{gameMode}/{endpoint}       -> data; text fields hold translation keys
  /{gameMode}/{endpoint}_{lang} -> {translation key: translated text}
"""
from __future__ import annotations

import json
import time
from typing import Any

import requests

from tarkov_loot_helper.Configuration.config import DATA_DIR

BASE_URL = "https://json.tarkov.dev"
CACHE_MAX_AGE = 60 * 60          # data/prices: refresh hourly
ENDPOINTS_MAX_AGE = 24 * 60 * 60  # endpoint list: refresh daily


class TarkovDevAPIConnector:
    def __init__(self, game_mode: str = "regular", language: str = "en",
                 cache_dir=DATA_DIR, max_age: int = CACHE_MAX_AGE):
        self.game_mode = game_mode
        self.language = language
        self.cache_dir = cache_dir
        self.max_age = max_age

    # ---- discovery ---------------------------------------------------------
    def endpoints(self, force: bool = False) -> dict:
        """{'endpoints': [...], 'gameModes': [...], 'languages': [...]}"""
        return self._get_cached("/endpoints", "endpoints", force, ENDPOINTS_MAX_AGE)["data"]

    def game_modes(self, force: bool = False) -> list[str]:
        return list(self.endpoints(force).get("gameModes") or [])

    def languages(self, force: bool = False) -> list[str]:
        return list(self.endpoints(force).get("languages") or [])

    # ---- data --------------------------------------------------------------
    def fetch(self, endpoint: str, translate: bool = True, force: bool = False) -> dict:
        """Fetch /{gameMode}/{endpoint}['data'], with translation keys replaced by text."""
        mode = self.game_mode
        data = self._get_cached(f"/{mode}/{endpoint}", f"{mode}_{endpoint}", force)["data"]
        if translate:
            path = f"/{mode}/{endpoint}_{self.language}"
            strings = self._get_cached(path, f"{mode}_{endpoint}_{self.language}", force)["data"]
            data = _translate(data, strings)
        return data

    def fetch_items(self, force: bool = False) -> dict[str, dict]:
        return self.fetch("items", force=force)["items"]

    def fetch_tasks(self, force: bool = False) -> dict[str, dict]:
        return self.fetch("tasks", force=force)["tasks"]

    def fetch_hideout(self, force: bool = False) -> dict[str, dict]:
        return self.fetch("hideout", force=force)

    def fetch_traders(self, force: bool = False) -> dict[str, dict]:
        return self.fetch("traders", force=force)

    # ---- transport ---------------------------------------------------------
    def get(self, path: str) -> dict:
        r = requests.get(BASE_URL + path, timeout=120)
        r.raise_for_status()
        return r.json()

    def _get_cached(self, path: str, name: str, force: bool, max_age: int | None = None) -> dict:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        file = self.cache_dir / f"{name}.json"
        age = self.max_age if max_age is None else max_age
        if not force and file.exists() and time.time() - file.stat().st_mtime < age:
            return json.loads(file.read_text(encoding="utf-8"))
        try:
            payload = self.get(path)
        except Exception:
            if file.exists():  # offline: fall back to stale cache
                return json.loads(file.read_text(encoding="utf-8"))
            raise
        file.write_text(json.dumps(payload), encoding="utf-8")
        return payload


def _translate(node: Any, strings: dict[str, str]) -> Any:
    if isinstance(node, str):
        return strings.get(node, node)
    if isinstance(node, dict):
        return {k: _translate(v, strings) for k, v in node.items()}
    if isinstance(node, list):
        return [_translate(v, strings) for v in node]
    return node
