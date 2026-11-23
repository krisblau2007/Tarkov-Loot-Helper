"""Downloads and caches item grid icons used as matching templates."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests

from .api import Item
from .config import ICON_DIR


def icon_path(item: Item):
    return ICON_DIR / f"{item.id}.webp"


def _download(item: Item) -> bool:
    path = icon_path(item)
    if path.exists() or not item.grid_image:
        return path.exists()
    try:
        r = requests.get(item.grid_image, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
        return True
    except requests.RequestException:
        return False


def prefetch(items: list[Item], progress=None) -> int:
    """Download all missing icons. Returns how many icons are available."""
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, result in enumerate(pool.map(_download, items), 1):
            ok += result
            if progress and i % 200 == 0:
                progress(i, len(items))
    return ok


def load_icon(item: Item) -> np.ndarray | None:
    path = icon_path(item)
    if not path.exists() and not _download(item):
        return None
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    return img
