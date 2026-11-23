"""Identifies equipped items (headwear, armor, weapons on sling/back, rig,
backpack...). These sit in slot panels rather than inventory grids, so they are
found by locating the slot panels and reading the item name Tarkov prints in
each panel's top-right corner."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tarkov_loot_helper.Application.api import Item
from tarkov_loot_helper.Identification.grid import Grid, line_masks
from tarkov_loot_helper.Identification.icons import load_icon
from tarkov_loot_helper.Identification.ocr import OcrLine, OcrWord, read_lines
from tarkov_loot_helper.Identification.textmatch import caliber_similarity, name_similarity, normalise

EQUIP_TYPES = {"gun", "armor", "backpack", "rig", "headphones", "helmet", "glasses", "wearable"}


@dataclass
class Phrase:
    text: str
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


@dataclass
class EquippedItem:
    item: Item
    score: float
    rect: tuple[int, int, int, int]


def phrases(lines: list[OcrLine]) -> list[Phrase]:
    """Split OCR lines into phrases: words separated by small gaps (OCR often joins
    the names of neighbouring tiles into one line)."""
    out: list[Phrase] = []
    for line in lines:
        group: list[OcrWord] = []
        for wd in sorted(line.words, key=lambda w: w.x):
            if group and wd.x - (group[-1].x + group[-1].w) > max(6, group[-1].h * 0.9):
                out.append(_merge(group))
                group = []
            group.append(wd)
        if group:
            out.append(_merge(group))
    return out


def _merge(words: list[OcrWord]) -> Phrase:
    x0 = min(w.x for w in words)
    y0 = min(w.y for w in words)
    x1 = max(w.x + w.w for w in words)
    y1 = max(w.y + w.h for w in words)
    return Phrase(" ".join(w.text for w in words), x0, y0, x1 - x0, y1 - y0)


def find_panels(bgr: np.ndarray, cell: float) -> list[tuple[int, int, int, int]]:
    horiz, vert = line_masks(bgr, cell)
    edges = cv2.dilate(horiz | vert, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 0.8 * cell <= w <= 8 * cell and 0.8 * cell <= h <= 6 * cell:
            rects.append((x, y, w, h))
    return rects


def _hist(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 2] > 40).astype(np.uint8)  # ignore the dark background
    h = cv2.calcHist([hsv], [0, 1], mask, [18, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def _inner_panels(panels: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Slot outlines are often found twice (outer + inner edge); keep the inner one."""
    kept: list[tuple[int, int, int, int]] = []
    for p in sorted(panels, key=lambda r: r[2] * r[3]):
        px, py, pw, ph = p
        dup = any(abs(px - k[0]) < 12 and abs(py - k[1]) < 12
                  and abs(pw - k[2]) < 24 and abs(ph - k[3]) < 24 for k in kept)
        if not dup:
            kept.append(p)
    return kept


def _read(bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int, scale: float = 3.0) -> str:
    crop = bgr[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return ""
    return " ".join(line.text for line in read_lines(crop, scale))


def _best_name_score(text: str, item: Item) -> float:
    """Name similarity, also trying without the first word (OCR noise at the left edge)."""
    score = name_similarity(text, item.short_name)
    parts = text.split(" ", 1)
    if len(parts) == 2 and len(normalise(parts[0])) <= 2 and len(normalise(parts[1])) >= 3:
        score = max(score, name_similarity(parts[1], item.short_name))
    return score


def detect_equipped(bgr: np.ndarray, lines: list[OcrLine], grids: list[Grid],
                    items: dict[str, Item], cell: float) -> list[EquippedItem]:
    candidates = [it for it in items.values()
                  if not it.is_preset and EQUIP_TYPES.intersection(it.types)]
    # Items that share a name with equipment but can't be equipped in a gear panel
    # (e.g. "USEC" dogtag vs "USEC" armband). Mods are excluded: they never sit in a
    # panel, but often share a name with their gun ("AK-104" muzzle brake).
    others = [it for it in items.values()
              if not it.is_preset and not EQUIP_TYPES.intersection(it.types) and "mods" not in it.types]
    found: list[EquippedItem] = []

    for panel in _inner_panels(find_panels(bgr, cell)):
        px, py, pw, ph_ = panel
        if any(g.contains(px + pw / 2, py + ph_ / 2) for g in grids):
            continue
        # The item's short name is printed in the panel's top-right corner.
        box = (px + pw // 4, py, px + pw, py + int(0.36 * cell))
        texts = [t for t in {_read(bgr, *box, scale=3.0), _read(bgr, *box, scale=2.0)}
                 if len(normalise(t)) >= 3]
        if not texts:
            continue
        scored = [(max(_best_name_score(t, it) for t in texts), it) for it in candidates]
        top = max((s for s, _ in scored), default=0.0)
        if top < 0.85:
            continue
        if max((_best_name_score(t, it) for it in others for t in texts), default=0.0) >= top:
            continue  # e.g. our own dogtag
        tied = [it for s, it in scored if s >= top - 0.02]

        crop = bgr[py:py + ph_, px:px + pw]
        if len(tied) > 1:
            if any(it.is_gun for it in tied):
                # guns: confirm with the caliber printed under the name
                cal = _read(bgr, px + pw // 3, py + int(0.25 * cell), px + pw, py + int(0.7 * cell))
                tied.sort(key=lambda it: caliber_similarity(cal, it.caliber) + (0.1 if it.is_gun else 0),
                          reverse=True)
            else:
                # same name, different variant (e.g. helmet colours): compare colours
                target = _hist(crop)
                def colour(it: Item) -> float:
                    icon = load_icon(it)
                    return -1.0 if icon is None else float(cv2.compareHist(target, _hist(icon), cv2.HISTCMP_CORREL))
                tied.sort(key=colour, reverse=True)
        found.append(EquippedItem(tied[0], top, panel))
    return found
