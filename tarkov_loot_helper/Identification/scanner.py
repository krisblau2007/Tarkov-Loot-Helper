"""Runs one full scan:
screenshot -> OCR -> grids -> item regions (footprint) -> identification,
plus equipped items in the gear slot panels."""
from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from tarkov_loot_helper.Application.api import Item
from tarkov_loot_helper.Configuration.config import Settings
from tarkov_loot_helper.Identification.equipment import detect_equipped
from tarkov_loot_helper.Identification.grid import Grid, Region, detect_grids, split_items
from tarkov_loot_helper.Identification.grid import find_square_slots
from tarkov_loot_helper.Identification.identify import Identifier
from tarkov_loot_helper.Identification.inspect_window import (find_windows, ignored_in_window, inside,
                                                              in_title_band, is_empty_slot_tile, is_slot_label)
from tarkov_loot_helper.Identification.ocr import OcrLine, read_lines

NAME_CONFIDENCE = 0.70  # min score for a name/caliber based identification


@dataclass
class Detection:
    rect: tuple[int, int, int, int]  # screen pixels
    width: int                        # footprint in cells
    height: int
    item: Item | None                 # confident identification (shown normally)
    guess: Item | None                # best guess, even if not confident
    score: float
    method: str                       # 'name', 'gun', 'icon' or 'equipped'
    kind: str = "grid"                # 'grid' or 'equipped'
    container: int = -1
    col: int = -1
    row: int = -1
    name_text: str = ""

    @property
    def value(self) -> int:
        return self.item.best_price if self.item else 0

    @property
    def price_per_slot(self) -> int:
        if not self.item:
            return 0
        if self.kind == "equipped":
            return self.item.price_per_slot
        return self.item.best_price // max(1, self.width * self.height)


@dataclass
class ScanResult:
    image: np.ndarray
    grids: list[Grid]
    detections: list[Detection]
    ocr: list[OcrLine]

    @property
    def identified(self) -> list[Detection]:
        return [d for d in self.detections if d.item is not None]

    @property
    def uncertain(self) -> list[Detection]:
        return [d for d in self.detections if d.item is None and d.guess is not None]


def _words_in(lines: list[OcrLine], x0, y0, x1, y1) -> str:
    words = [w for line in lines for w in line.words
             if x0 <= w.x + w.w / 2 <= x1 and y0 <= w.y + w.h / 2 <= y1]
    words.sort(key=lambda w: w.x)
    return " ".join(w.text for w in words)


def _region_text(lines: list[OcrLine], reg: Region, grid: Grid) -> tuple[str, str]:
    x, y, w, h = reg.rect
    band = 0.32 * grid.pitch_y
    name = _words_in(lines, x, y, x + w, y + band)
    caliber = _words_in(lines, x, y + h - band, x + 0.7 * w, y + h)
    return name, caliber


def _region_capacity(lines: list[OcrLine], reg: Region, grid: Grid, bgr: np.ndarray) -> int:
    """Magazine size from the "rounds/capacity" counter in the tile's bottom-right
    corner (e.g. "0/17" -> 17). Returns 0 if there is no counter."""
    x, y, w, h = reg.rect
    band = 0.32 * grid.pitch_y
    text = _words_in(lines, x + 0.3 * w, y + h - band, x + w, y + h)
    if "/" not in text:  # the full-screen pass may have missed it: re-read magnified
        strip = bgr[int(y + h - band):y + h, int(x + 0.3 * w):x + w]
        if strip.size:
            text = " ".join(line.text for line in read_lines(strip, 3.0))
    # OCR reads Tarkov's slashed zero as "Ø" (Ø); fold look-alikes to digits
    text = text.upper().replace("Ø", "0").replace("O", "0").replace("I", "1").replace("L", "1")
    m = re.search(r"(\d{1,3})\s*/\s*(\d{1,3})", text)
    return int(m.group(2)) if m else 0


def _drop_quick_use_bar(grids: list[Grid], lines: list[OcrLine] = (), cell: float = 84.0) -> list[Grid]:
    """The Quick Use bar is a row of shortcut slots. They mirror items stored
    elsewhere, so scanning them would only produce duplicates. It is found by its
    "QUICK USE" label (slots start just below it), or as a row of 6+ 1x1 slots."""
    from tarkov_loot_helper.Identification.textmatch import normalise
    drop: set[int] = set()
    for line in lines:
        if normalise(line.text) == normalise("QUICK USE"):
            lx, ly, lw, lh = line.rect
            for g in grids:
                gx, gy, gw, gh = g.rect
                if ly < gy < ly + lh + 1.5 * cell and lx - cell < gx < lx + 14 * cell and g.cols * g.rows <= 2:
                    drop.add(id(g))
    singles = [g for g in grids if g.cols == 1 and g.rows == 1]
    for g in singles:
        row = [o for o in singles if abs(o.y0 - g.y0) <= 4]
        if len(row) >= 6:
            drop.update(id(o) for o in row)
    return [g for g in grids if id(g) not in drop]


def _without_boxes(lines: list[OcrLine], boxes) -> list[OcrLine]:
    """OCR lines with every word inside one of the boxes removed."""
    if not boxes:
        return lines
    out = []
    for line in lines:
        words = [w for w in line.words
                 if not any(bx <= w.x + w.w / 2 <= bx + bw and by <= w.y + w.h / 2 <= by + bh
                            for bx, by, bw, bh in boxes)]
        if words:
            out.append(OcrLine(" ".join(w.text for w in words), words))
    return out


def _unplaced_hover(detections: list[Detection], tips: list, blacklist: set, cell: float) -> list[Detection]:
    """A hovered item whose slot couldn't be found (its highlight can hide a whole
    special/pocket slot) is still known from the tooltip: add it just left of it."""
    extra = []
    for tip in tips:
        if tip.item.id in blacklist or any(d.method == "tooltip" and d.item is tip.item for d in detections):
            continue
        bx, by, bw, bh = tip.box
        # Already found by its own tile (e.g. tooltip opened to the left)?
        if any((d.item or d.guess) is not None and (d.item or d.guess).short_name == tip.item.short_name
               and bx - 3 * cell <= d.rect[0] + d.rect[2] / 2 <= bx + bw + 3 * cell
               and by - 3 * cell <= d.rect[1] + d.rect[3] / 2 <= by + bh + 3 * cell for d in detections):
            continue
        rect = (int(bx - cell - 8), int(by - cell / 2), int(cell), int(cell))
        cx, cy = rect[0] + cell / 2, rect[1] + cell / 2
        if any(d.rect[0] <= cx <= d.rect[0] + d.rect[2] and d.rect[1] <= cy <= d.rect[1] + d.rect[3]
               for d in detections):
            continue  # that spot is already another item: the cursor isn't there
        extra.append(Detection(rect, 1, 1, tip.item, tip.item, 1.0, "tooltip", "grid", name_text=tip.item.short_name))
    return extra


@dataclass
class Tooltip:
    item: Item
    text_rect: tuple[int, int, int, int]
    box: tuple[int, int, int, int]  # the whole tooltip panel (hides what's under it)


def _tooltip_candidates(lines: list[OcrLine]) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Each OCR line, plus each line joined with the line directly below it: long
    names wrap onto a second line ("MTEK STRIKE Ballistic helmet" /
    "(Coyote/MultiCam Arid)")."""
    out = [(l.text, l.rect) for l in lines]
    for a in lines:
        ax, ay, aw, ah = a.rect
        for b in lines:
            bx, by, bw, bh = b.rect
            if b is not a and abs(bx - ax) <= 12 and 0 < by - (ay + ah) <= 1.2 * ah:
                x0, y0 = min(ax, bx), ay
                rect = (x0, y0, max(ax + aw, bx + bw) - x0, by + bh - y0)
                out.append((a.text + " " + b.text, rect))
    return out


def _find_tooltips(lines: list[OcrLine], identifier: Identifier, windows=(), cell: float = 84.0) -> list[Tooltip]:
    """When the mouse rests on an item, Tarkov shows its full name in a tooltip next
    to the cursor. Find OCR text that is an item's full name (skipping inspect
    window titles, which also show full names)."""
    from difflib import SequenceMatcher

    from tarkov_loot_helper.Identification.textmatch import normalise
    found: list[tuple[float, Tooltip]] = []
    for text, rect in _tooltip_candidates(lines):
        cx, cy = rect[0] + rect[2] / 2, rect[1] + rect[3] / 2
        if any(in_title_band(b, rect[0] + 4, rect[1] + 4, cell) for b in windows):
            continue
        if len(normalise(text)) < 10:
            continue
        best, best_s, second = None, 0.0, 0.0
        key = normalise(text)
        words = {normalise(w) for w in text.split() if len(normalise(w)) >= 2}
        if len(words) < 2:
            continue
        for it in identifier.items.values():
            if it.is_preset:
                continue
            name = normalise(it.name)
            if abs(len(name) - len(key)) > 12:  # OCR may drop a word or two
                continue
            # character similarity + share of the tooltip's words found in the name
            name_words = {normalise(w) for w in it.name.split()}
            recall = len(words & name_words) / len(words) if words else 0.0
            ratio = SequenceMatcher(None, key, name).ratio()
            if recall < 0.6 or ratio < 0.7:
                continue
            s = 0.5 * ratio + 0.5 * recall
            if s > best_s:
                best, best_s, second = it, s, best_s
            elif s > second:
                second = s
        if best is None or best_s < 0.8 or best_s - second < 0.03:
            continue
        lx, ly, lw, lh = rect
        found.append((best_s, Tooltip(best, rect, (lx - 32, ly - 20, lw + 64, lh + 40))))
    # A one-line match and its two-line version describe the same tooltip: keep the
    # better-scoring one of any overlapping pair.
    found.sort(key=lambda t: -t[0])
    tips: list[Tooltip] = []
    for _, tip in found:
        x, y, w, h = tip.text_rect
        if any(min(x + w, t.text_rect[0] + t.text_rect[2]) - max(x, t.text_rect[0]) > 0
               and min(y + h, t.text_rect[1] + t.text_rect[3]) - max(y, t.text_rect[1]) > 0 for t in tips):
            continue
        tips.append(tip)
    return tips


def _hovered_placement(tip: Tooltip, grids: list[Grid], lines: list[OcrLine],
                       bgr: np.ndarray) -> tuple[int, tuple[int, int, int, int]] | None:
    """Where the hovered item sits: (grid index, pixel rect). The cursor is just
    above-left of the tooltip; the item covers the cursor's cell and has the
    tooltip item's size. Of the possible placements, prefer the one whose
    top-right label matches the item's short name, then the brightest (hover
    highlight). Guns are skipped: their size depends on attachments."""
    from tarkov_loot_helper.Identification.textmatch import name_similarity
    it = tip.item
    if it.is_gun:
        return None
    bx, by, bw, bh = tip.box
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    best = None
    # The tooltip opens next to the cursor - below-right normally, but above it near
    # the bottom of the screen, left of it near the right edge, and clamped to the
    # screen. So consider every cell within ~1.5 cells of the tooltip.
    margin = 1.5 * max((g.pitch_x for g in grids), default=84.0)
    for gi, g in enumerate(grids):
        cells = [(c, r) for r in range(g.rows) for c in range(g.cols)
                 if bx - margin <= g.x0 + (c + 0.5) * g.pitch_x <= bx + bw + margin
                 and by - margin <= g.y0 + (r + 0.5) * g.pitch_y <= by + bh + margin]
        for c, r in cells:
            for w, h in {(it.width, it.height), (it.height, it.width)}:
                for dc in range(w):
                    for dr in range(h):
                        c0, r0 = c - dc, r - dr
                        if c0 < 0 or r0 < 0 or c0 + w > g.cols or r0 + h > g.rows:
                            continue
                        x, y = int(g.x0 + c0 * g.pitch_x), int(g.y0 + r0 * g.pitch_y)
                        rect = (x, y, int(w * g.pitch_x), int(h * g.pitch_y))
                        label = _words_in(lines, x, y, x + rect[2], y + 0.32 * g.pitch_y)
                        score = name_similarity(label, it.short_name) if label else 0.0
                        score += float(gray[y:y + rect[3], x:x + rect[2]].mean()) / 1000.0
                        if best is None or score > best[0]:
                            best = (score, gi, rect)
    # Only trust a placement whose label matches the item's short name.
    if best and best[0] >= 0.75:
        return best[1], best[2]
    return None


def _apply_tooltips(detections: list[Detection], hovered: list, cell: float) -> None:
    """The tooltip's full name is unambiguous, so it overrides whatever the hovered
    tile was identified as. The hovered tile is the placement found earlier, or
    (for guns) the tile whose right edge is just left of the tooltip."""
    from tarkov_loot_helper.Identification.textmatch import name_similarity
    for tip, placement in hovered:
        if placement:
            px, py, _, _ = placement[1]
            cands = [d for d in detections if d.kind == "grid" and abs(d.rect[0] - px) <= 4 and abs(d.rect[1] - py) <= 4]
        else:
            bx, by, _, _ = tip.box
            # Only a tile whose own label is unreadable or agrees with the tooltip:
            # otherwise the tooltip belongs to some other tile (e.g. it opened leftwards).
            cands = [d for d in detections if d.kind == "grid"
                     and bx - 1.2 * cell <= d.rect[0] + d.rect[2] <= bx + 12
                     and d.rect[1] - cell <= by <= d.rect[1] + d.rect[3] + 4
                     and (len(d.name_text.strip()) < 2 or name_similarity(d.name_text, tip.item.short_name) >= 0.6)]
        if not cands:
            continue
        bx, by = tip.box[0], tip.box[1]
        d = min(cands, key=lambda d: abs(bx - (d.rect[0] + d.rect[2])) + abs(by - d.rect[1]) * 0.2)
        d.item = d.guess = tip.item
        d.score, d.method = 1.0, "tooltip"


def _drop_duplicate_windows(detections: list[Detection]) -> list[Detection]:
    """An opened container shows its contents twice: in a floating window and in
    its slot. Drop a grid whose items and layout exactly repeat another grid's."""
    by_grid: dict[int, list[tuple]] = {}
    for d in detections:
        if d.kind == "grid":
            by_grid.setdefault(d.container, []).append((d.col, d.row, d.width, d.height,
                                                         (d.item or d.guess).id if (d.item or d.guess) else None))
    seen: dict[tuple, int] = {}
    drop: set[int] = set()
    for gi in sorted(by_grid):
        sig = tuple(sorted(by_grid[gi]))
        if len(sig) >= 2 and sig in seen:
            drop.add(gi)
        else:
            seen[sig] = gi
    return [d for d in detections if not (d.kind == "grid" and d.container in drop)]


def scan_image(bgr: np.ndarray, settings: Settings, identifier: Identifier) -> ScanResult:
    cell = settings.cell_px(bgr.shape[0])
    blacklist = set(settings.blacklist)
    lines = read_lines(bgr, scale=2.0)
    grids, _ = detect_grids(bgr, cell)
    grids = _drop_quick_use_bar(grids, lines, cell)
    # Inspect windows: their attachment slots are separate 1x1 squares, found
    # separately; whatever lies under a window is hidden.
    windows = find_windows(lines, cell)
    for box in windows:
        # drop grids lying entirely inside the window (they are its slot rows); grids
        # merely behind it (the stash) stay, with their hidden part ignored below
        grids = [g for g in grids if not (inside(box, g.x0 + 2, g.y0 + 2)
                                          and inside(box, g.x0 + g.cols * g.pitch_x - 2, g.y0 + g.rows * g.pitch_y - 2))]
        grids += find_square_slots(bgr, cell, box)
    tips = _find_tooltips(lines, identifier, windows, cell)
    # Tooltip text lies on top of other tiles; never read it as their names.
    tile_lines = _without_boxes(lines, [t.box for t in tips])
    hovered = [(t, _hovered_placement(t, grids, tile_lines, bgr)) for t in tips]
    regions = split_items(bgr, grids, [t.box for t in tips] + windows, [p for _, p in hovered if p], cell)

    detections: list[Detection] = []
    for reg in regions:
        x, y, w, h = reg.rect
        crop = bgr[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        grid = grids[reg.grid_index]
        cx, cy = x + w / 2, y + h / 2
        in_window = any(inside(b, cx, cy) for b in windows)
        if in_window and not grid.slot:
            continue  # the part of a grid hidden behind an inspect window
        name_text, caliber_text = _region_text(tile_lines, reg, grid)
        if in_window and (is_slot_label(name_text) or is_empty_slot_tile(crop)):
            continue  # empty attachment slot (or built-in armor part)
        if len(name_text.strip()) < 2 and not reg.looks_empty:
            # The full-screen pass missed it: re-read just this tile's name strip, magnified.
            strip = bgr[y:y + int(0.32 * grid.pitch_y), x + w // 5:x + w]
            for scale in (3.0, 4.0, 2.0) if strip.size else ():  # OCR is fussy about size
                name_text = " ".join(line.text for line in read_lines(strip, scale))
                if len(name_text.strip()) >= 2:
                    break
        has_text = len(name_text.strip()) >= 2
        # Is there an item at all? Empty slots are a flat see-through background;
        # filled ones have an icon on a coloured tile.
        if (reg.looks_empty or reg.unsearched) and not has_text:
            continue  # empty slot, or a loot cell not searched yet
        occluded = any(min(x + w, bx + bw) - max(x, bx) > 4 and min(y + h, by + bh) - max(y, by) > 4
                       for bx, by, bw, bh in (t.box for t in tips))
        capacity = _region_capacity(tile_lines, reg, grid, bgr) if not reg.looks_empty else 0
        guess, score, method = identifier.identify_region(crop, reg.width, reg.height, name_text, caliber_text,
                                                          occluded, capacity, extend=occluded)
        if guess is None:
            continue
        if guess.id in blacklist:
            continue
        if in_window and ignored_in_window(guess):
            continue  # mounts/rails carrying other parts, built-in armor parts
        if method in ("name", "gun"):
            confident = score >= NAME_CONFIDENCE
        else:
            confident = score >= settings.min_confidence and not reg.looks_empty
            if not confident and not has_text and score < 0.6 and (reg.framed < 0.75 or score < 0.3):
                continue  # nothing readable, no convincing icon, not clearly a tile: skip
            # (a clearly framed tile is kept as a "?" guess so the user sees it's there)
        if not confident and score >= 0.6 and not reg.looks_empty and guess.best_price < settings.low_value_threshold:
            confident = True  # a cheap item: being slightly wrong costs little
        detections.append(Detection(reg.rect, reg.width, reg.height, guess if confident else None, guess,
                                    score, method, "grid", reg.grid_index, reg.col, reg.row, name_text))

    _apply_tooltips(detections, hovered, cell)
    detections = _drop_duplicate_windows(detections)
    detections += _unplaced_hover(detections, tips, blacklist, cell)

    for eq in detect_equipped(bgr, lines, grids, identifier.items, cell):
        if eq.item.id in blacklist:
            continue
        ex, ey, ew, eh = eq.rect
        if any(inside(b, ex + ew / 2, ey + eh / 2) for b in windows):
            continue  # a gear panel behind (or the picture inside) an inspect window
        # its name label (top-right strip) partly under a window: the name may be cut
        lx0, lx1, ly0, ly1 = ex + ew * 0.25, ex + ew, ey, ey + 0.36 * cell
        if any(min(lx1, b[0] + b[2]) > max(lx0, b[0]) and min(ly1, b[1] + b[3]) > max(ly0, b[1]) for b in windows):
            continue
        detections.append(Detection(eq.rect, eq.item.width, eq.item.height, eq.item, eq.item,
                                    eq.score, "equipped", "equipped", name_text=eq.item.short_name))
    return ScanResult(bgr, grids, detections, lines)


def render_debug(result: ScanResult) -> np.ndarray:
    """Annotated copy of the screenshot, useful for tuning UI scale / thresholds.
    Orange = grid, green = identified, yellow = equipped, red = uncertain guess."""
    out = result.image.copy()
    for g in result.grids:
        x, y, w, h = g.rect
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 160, 0), 2)
        cv2.putText(out, f"{g.cols}x{g.rows}", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 160, 0), 1)
    for d in result.detections:
        x, y, w, h = d.rect
        if d.kind == "equipped":
            color = (0, 220, 255)
        else:
            color = (0, 220, 0) if d.item else (0, 0, 255)
        cv2.rectangle(out, (x + 2, y + 2), (x + w - 2, y + h - 2), color, 1)
        it = d.item or d.guess
        label = f"{'' if d.item else '?'}{it.short_name} {d.score:.2f} {d.method[0]}"
        cv2.putText(out, label, (x + 4, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
    return out
