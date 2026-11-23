"""Runs one full scan: screenshot -> grids -> item regions -> identifications."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .api import Item
from .config import Settings
from .grid import Grid, Region, detect_grids, split_items
from .identify import Identifier


@dataclass
class Detection:
    region: Region
    item: Item | None
    score: float
    rotated: bool

    @property
    def confident(self) -> bool:
        return self.item is not None


@dataclass
class ScanResult:
    image: np.ndarray
    grids: list[Grid]
    detections: list[Detection]

    @property
    def identified(self) -> list[Detection]:
        return [d for d in self.detections if d.item is not None]


def scan_image(bgr: np.ndarray, settings: Settings, identifier: Identifier) -> ScanResult:
    cell = settings.cell_px(bgr.shape[0])
    grids, masks = detect_grids(bgr, cell)
    regions = split_items(bgr, grids, masks)
    detections = []
    for reg in regions:
        x, y, w, h = reg.rect
        crop = bgr[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        item, score, rotated = identifier.identify(crop, reg.width, reg.height)
        if score < settings.min_confidence:
            item = None
        detections.append(Detection(reg, item, score, rotated))
    return ScanResult(bgr, grids, detections)


def render_debug(result: ScanResult) -> np.ndarray:
    """Annotated copy of the screenshot, useful for tuning UI scale / thresholds."""
    out = result.image.copy()
    for g in result.grids:
        x, y, w, h = g.rect
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 160, 0), 2)
        cv2.putText(out, f"{g.cols}x{g.rows}", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 160, 0), 1)
    for d in result.detections:
        x, y, w, h = d.region.rect
        color = (0, 220, 0) if d.item else (0, 0, 255)
        cv2.rectangle(out, (x + 2, y + 2), (x + w - 2, y + h - 2), color, 1)
        label = f"{d.item.short_name} {d.score:.2f}" if d.item else f"? {d.score:.2f}"
        cv2.putText(out, label, (x + 4, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
    return out
