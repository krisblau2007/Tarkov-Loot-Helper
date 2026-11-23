"""Text recognition using the OCR engine built into Windows 10/11 (Windows.Media.Ocr).

No extra install is needed beyond the winrt Python bindings in requirements.txt;
it uses the OCR language packs already present for the user's Windows language.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import cv2
import numpy as np
from winrt.windows.graphics.imaging import BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.storage.streams import DataWriter


@dataclass
class OcrWord:
    text: str
    x: int
    y: int
    w: int
    h: int


@dataclass
class OcrLine:
    text: str
    words: list[OcrWord]

    @property
    def rect(self) -> tuple[int, int, int, int]:
        x0 = min(w.x for w in self.words)
        y0 = min(w.y for w in self.words)
        x1 = max(w.x + w.w for w in self.words)
        y1 = max(w.y + w.h for w in self.words)
        return x0, y0, x1 - x0, y1 - y0


_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = OcrEngine.try_create_from_user_profile_languages()
        if _engine is None:
            raise RuntimeError("Windows OCR is unavailable: install an OCR language pack "
                               "(Settings > Time & language > Language).")
    return _engine


def _to_bitmap(bgr: np.ndarray) -> SoftwareBitmap:
    bgra = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA))
    writer = DataWriter()
    writer.write_bytes(bgra.tobytes())
    bmp = SoftwareBitmap(BitmapPixelFormat.BGRA8, bgra.shape[1], bgra.shape[0], BitmapAlphaMode.PREMULTIPLIED)
    bmp.copy_from_buffer(writer.detach_buffer())
    return bmp


def read_lines(bgr: np.ndarray, scale: float = 2.0) -> list[OcrLine]:
    """OCR an image. Returned coordinates are in the input image's pixels."""
    engine = _get_engine()
    img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else bgr
    max_dim = OcrEngine.max_image_dimension
    if max(img.shape[:2]) > max_dim:  # tile very large images
        return _read_tiled(bgr, scale, max_dim)
    result = asyncio.run(_recognize(engine, _to_bitmap(img)))
    lines = []
    for line in result.lines:
        words = [OcrWord(w.text, int(w.bounding_rect.x / scale), int(w.bounding_rect.y / scale),
                         int(w.bounding_rect.width / scale) + 1, int(w.bounding_rect.height / scale) + 1)
                 for w in line.words]
        if words:
            lines.append(OcrLine(line.text, words))
    return lines


def _read_tiled(bgr: np.ndarray, scale: float, max_dim: int) -> list[OcrLine]:
    tile = int(max_dim / scale) - 64
    out: list[OcrLine] = []
    h, w = bgr.shape[:2]
    for y in range(0, h, tile - 64):
        for x in range(0, w, tile - 64):
            for line in read_lines(bgr[y:y + tile, x:x + tile], scale):
                for wd in line.words:
                    wd.x += x
                    wd.y += y
                out.append(line)
    return out


async def _recognize(engine, bmp):
    return await engine.recognize_async(bmp)
