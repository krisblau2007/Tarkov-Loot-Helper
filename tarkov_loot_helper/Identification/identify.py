"""Identifies an item region by comparing it against tarkov.dev grid icons of
the same footprint (width x height, or rotated)."""
from __future__ import annotations

import cv2
import numpy as np

from tarkov_loot_helper.Application.api import Item
from tarkov_loot_helper.Configuration.config import DATA_DIR
from tarkov_loot_helper.Identification.icons import icon_path, load_icon

TEMPLATE_DIR = DATA_DIR / "templates"

S = 24  # normalised pixels per cell for comparison


def _mask(w: int, h: int) -> np.ndarray:
    """Ignore the in-game overlays: short name (top-right) and stack count (bottom-right)."""
    m = np.ones((h * S, w * S), bool)
    m[: int(S * 0.3), int(w * S * 0.35):] = False
    m[h * S - int(S * 0.3):, int(w * S * 0.5):] = False
    return m


def _normalise(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w * S, h * S), interpolation=cv2.INTER_AREA).astype(np.float32)


class Identifier:
    def __init__(self, items: dict[str, Item]):
        self.items = items
        self._bank: dict[tuple[int, int], tuple[list[tuple[Item, bool]], np.ndarray]] = {}

    def _templates(self, w: int, h: int):
        """Stack of normalised templates for footprint w x h (incl. rotated items)."""
        key = (w, h)
        if key in self._bank:
            return self._bank[key]
        candidates = []
        for it in self.items.values():
            for rotated in (False, True):
                iw, ih = (it.height, it.width) if rotated else (it.width, it.height)
                if (iw, ih) != (w, h) or (rotated and it.width == it.height):
                    continue
                candidates.append((it, rotated))

        # Reuse the prepared templates from disk if the item set hasn't changed.
        cache_file = TEMPLATE_DIR / f"{w}x{h}_s{S}.npz"
        wanted = [f"{it.id}:{int(rot)}" for it, rot in candidates if icon_path(it).exists()]
        if cache_file.exists():
            try:
                cached = np.load(cache_file)
                keys = list(cached["keys"])
                if set(keys) >= set(wanted):
                    lookup = {f"{it.id}:{int(rot)}": (it, rot) for it, rot in candidates}
                    pick = [i for i, k in enumerate(keys) if k in lookup]
                    refs = [lookup[keys[i]] for i in pick]
                    self._bank[key] = (refs, cached["arr"][pick])
                    return self._bank[key]
            except Exception:  # noqa: BLE001 - corrupt cache, rebuild
                pass

        refs, imgs = [], []
        for it, rotated in candidates:
            icon = load_icon(it)
            if icon is None:
                continue
            if rotated:
                icon = cv2.rotate(icon, cv2.ROTATE_90_CLOCKWISE)
            refs.append((it, rotated))
            imgs.append(_normalise(icon, w, h))
        mask = _mask(w, h)
        if imgs:
            arr = np.stack(imgs)[:, mask]  # (N, P, 3)
            arr = arr.reshape(len(imgs), -1)
            arr -= arr.mean(axis=1, keepdims=True)
            arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-6
        else:
            arr = np.zeros((0, int(mask.sum()) * 3), np.float32)
        self._bank[key] = (refs, arr)
        if refs:
            TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
            keys = np.array([f"{it.id}:{int(rot)}" for it, rot in refs])
            np.savez(cache_file, keys=keys, arr=arr.astype(np.float32))
        return self._bank[key]

    def identify(self, crop: np.ndarray, w: int, h: int) -> tuple[Item | None, float, bool]:
        refs, bank = self._templates(w, h)
        if not refs:
            return None, 0.0, False
        v = _normalise(crop, w, h)[_mask(w, h)].reshape(-1)
        v -= v.mean()
        v /= np.linalg.norm(v) + 1e-6
        scores = bank @ v  # Pearson correlation against every template
        best = int(np.argmax(scores))
        item, rotated = refs[best]
        return item, float(scores[best]), rotated
