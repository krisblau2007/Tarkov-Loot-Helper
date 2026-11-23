"""Fuzzy matching of OCR text against item short names and calibers."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

# Characters OCR commonly confuses are folded to one canonical form (applied to both sides).
_FOLD = str.maketrans({
    "Ø": "0", "ø": "0", "O": "0", "Q": "0", "D": "0",
    "S": "5", "I": "1", "L": "1", "|": "1", "!": "1", "B": "8", "Z": "2",
})


def normalise(text: str) -> str:
    text = text.upper().translate(_FOLD)
    return re.sub(r"[^0-9A-Z]", "", text)


def name_similarity(ocr: str, short_name: str) -> float:
    """0..1 similarity. Handles names Tarkov truncates to fit the tile (e.g. 'Moonshin')."""
    a, b = normalise(ocr), normalise(short_name)
    if not a or not b:
        return 0.0
    full = SequenceMatcher(None, a, b).ratio()
    if len(a) < len(b) and len(a) >= 4:
        weight = 0.9 + 0.1 * len(a) / len(b)
        prefix = SequenceMatcher(None, a, b[:len(a)]).ratio() * weight  # truncated by Tarkov
        suffix = SequenceMatcher(None, a, b[-len(a):]).ratio() * weight  # start hidden by a badge
        full = max(full, prefix, suffix)
    return full


def caliber_similarity(ocr: str, caliber: str) -> float:
    a, b = normalise(ocr), normalise(caliber)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()
