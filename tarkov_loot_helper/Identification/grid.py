"""Detects inventory grids (stash, backpack, rig, pockets, containers...) and
splits them into item regions.

Approach:
  1. Find thin grid lines that are brighter than their surroundings.
  2. Keep only long horizontal / vertical runs (at least ~60% of a cell).
  3. Grid-line crossings become lattice points; points one cell apart that are
     joined by a visible line are linked. Each connected lattice is one grid.
  4. Every border between two cells is measured. Item tiles draw a bright frame
     around themselves, while plain grid lines stay dim, even when they are
     visible through an item's image. Cells joined by dim borders are merged, so
     each item becomes one rectangle and its footprint (cols x rows) is known
     before identification starts.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

LINE_HIGHPASS = 3      # how much brighter than its surroundings a grid line must be
BORDER_BRIGHT = 62     # grey level separating item frames (~80-140) from grid lines (~30-50)
EMPTY_TEXTURE = 10     # regions with less internal edge strength than this are empty slots
BORDER_STEP = 18       # brightness jump between two flat tile backgrounds that counts as a border
BORDER_RIDGE = 15      # how much brighter a frame is than the pixels 3px either side


@dataclass
class Grid:
    x0: float
    y0: float
    pitch_x: float
    pitch_y: float
    cols: int
    rows: int
    slot: bool = False  # a single-item slot found by _long_slots (never merged)

    def cell_rect(self, c: int, r: int) -> tuple[int, int, int, int]:
        x = self.x0 + c * self.pitch_x
        y = self.y0 + r * self.pitch_y
        return int(round(x)), int(round(y)), int(round(self.pitch_x)), int(round(self.pitch_y))

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (int(self.x0), int(self.y0),
                int(round(self.cols * self.pitch_x)), int(round(self.rows * self.pitch_y)))

    def contains(self, x: float, y: float) -> bool:
        gx, gy, gw, gh = self.rect
        return gx <= x <= gx + gw and gy <= y <= gy + gh


@dataclass
class Region:
    grid_index: int
    col: int
    row: int
    width: int   # in cells
    height: int  # in cells
    rect: tuple[int, int, int, int]  # x, y, w, h in image pixels
    framed: float = 0.0   # fraction of the outline that is a bright item frame
    looks_empty: bool = False
    unsearched: bool = False  # striped "not searched yet" cell of a loot container


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def line_masks(bgr: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    highpass = gray.astype(np.int16) - cv2.blur(gray, (7, 7)).astype(np.int16)
    mask = (highpass > LINE_HIGHPASS).astype(np.uint8)
    k = max(8, int(cell * 0.6))
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, k), np.uint8))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, 1), np.uint8))
    horiz = cv2.dilate(horiz, np.ones((3, 1), np.uint8))
    vert = cv2.dilate(vert, np.ones((1, 3), np.uint8))
    return horiz, vert


def _segment_present(mask: np.ndarray, x1, y1, x2, y2, thresh=0.5) -> bool:
    n = int(max(abs(x2 - x1), abs(y2 - y1)))
    if n < 4:
        return False
    t = np.linspace(0.15, 0.85, max(4, n // 2))
    xs = np.clip(np.round(x1 + (x2 - x1) * t).astype(int), 0, mask.shape[1] - 1)
    ys = np.clip(np.round(y1 + (y2 - y1) * t).astype(int), 0, mask.shape[0] - 1)
    return mask[ys, xs].mean() >= thresh


def _fit_axis(values: np.ndarray, cell: float) -> tuple[float, float, int]:
    """Robustly fit evenly spaced lines to point coordinates -> (origin, pitch, count)."""
    base = values.min()
    idx = np.round((values - base) / cell).astype(int)
    centers = {i: float(np.median(values[idx == i])) for i in np.unique(idx)}
    keys = np.array(sorted(centers))
    vals = np.array([centers[k] for k in keys])
    if len(keys) >= 2:
        pitch, origin = np.polyfit(keys, vals, 1)
    else:
        pitch, origin = cell, vals[0]
    return float(origin + keys[0] * pitch), float(pitch), int(keys[-1] - keys[0])


def detect_grids(bgr: np.ndarray, cell: float) -> tuple[list[Grid], tuple[np.ndarray, np.ndarray]]:
    horiz, vert = line_masks(bgr, cell)
    cross = cv2.dilate(horiz, np.ones((3, 3), np.uint8)) & cv2.dilate(vert, np.ones((3, 3), np.uint8))
    _, _, _, cents = cv2.connectedComponentsWithStats(cross, connectivity=8)
    pts = cents[1:]
    if len(pts) < 4:
        return [], (horiz, vert)

    tol = max(2.0, cell * 0.08)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(pts):
        buckets.setdefault((int(x // cell), int(y // cell)), []).append(i)

    uf = _UF(len(pts))
    linked = np.zeros(len(pts), bool)
    for i, (x, y) in enumerate(pts):
        bx, by = int(x // cell), int(y // cell)
        for dx in (0, 1, 2):
            for dy in (-1, 0, 1, 2):
                for j in buckets.get((bx + dx, by + dy), ()):
                    if j <= i:
                        continue
                    x2, y2 = pts[j]
                    ddx, ddy = x2 - x, y2 - y
                    if abs(abs(ddx) - cell) <= tol and abs(ddy) <= tol:
                        ok = _segment_present(horiz, x, y, x2, y2)
                    elif abs(abs(ddy) - cell) <= tol and abs(ddx) <= tol:
                        ok = _segment_present(vert, x, y, x2, y2)
                    else:
                        continue
                    if ok:
                        uf.union(i, j)
                        linked[i] = linked[j] = True

    groups: dict[int, list[int]] = {}
    for i in np.flatnonzero(linked):
        groups.setdefault(uf.find(i), []).append(i)

    grids: list[Grid] = []
    for members in groups.values():
        if len(members) < 4:
            continue
        g = pts[members]
        x0, px, cols = _fit_axis(g[:, 0], cell)
        y0, py, rows = _fit_axis(g[:, 1], cell)
        if cols < 1 or rows < 1:
            continue
        grids.append(Grid(x0, y0, px, py, cols, rows))
    grids += _long_slots(pts, grids, cell, horiz, vert, tol)
    grids = _merge_aligned(grids, pts, cell, (horiz, vert))
    grids = [_trim_walls(g, (horiz, vert), max(3.0, cell * 0.05)) for g in grids]
    grids.sort(key=lambda gr: (gr.y0, gr.x0))
    return grids, (horiz, vert)


def _long_slots(pts: np.ndarray, grids: list[Grid], cell: float, horiz, vert, tol: float) -> list[Grid]:
    """Rig/pocket slots holding a single 1x2 or 1x3 item (e.g. a magazine) have no
    crossings in the middle, so the lattice search can't see them. Find them as
    rectangles 1 cell wide and 2-3 cells long whose four sides are unbroken lines."""
    found: list[Grid] = []
    # Slot borders are often doubled (slot outline + the item's own frame a few px in),
    # so allow the side check a few px of slack across the line.
    horiz_w = cv2.dilate(horiz, np.ones((5, 1), np.uint8))
    vert_w = cv2.dilate(vert, np.ones((1, 5), np.uint8))

    def near(x, y):
        d = np.abs(pts[:, 0] - x) + np.abs(pts[:, 1] - y)
        k = int(np.argmin(d))
        return pts[k] if d[k] <= 2 * tol else None

    def overlaps(r, others) -> bool:
        x, y, w, h = r
        for g in others:
            gx, gy, gw, gh = g.rect
            ix = min(x + w, gx + gw) - max(x, gx)
            iy = min(y + h, gy + gh) - max(y, gy)
            if ix > 6 and iy > 6:
                return True
        return False

    for x, y in pts:
        for kx, ky in ((1, 2), (1, 3), (2, 1), (3, 1)):
            tr = near(x + kx * cell, y)
            bl = near(x, y + ky * cell)
            if tr is None or bl is None:
                continue
            br = near(tr[0], bl[1])
            if br is None:
                continue
            g = Grid(float(x), float(y), (tr[0] - x) / kx, (bl[1] - y) / ky, kx, ky, slot=True)
            if overlaps(g.rect, grids + found):
                continue
            # the long sides must be clear; the short ends may be partly covered by the
            # item's counter or art
            th, tv = (0.85, 0.6) if kx > ky else (0.6, 0.85)
            sides = (_segment_present(horiz_w, x, y, tr[0], tr[1], th)
                     and _segment_present(horiz_w, bl[0], bl[1], br[0], br[1], th)
                     and _segment_present(vert_w, x, y, bl[0], bl[1], tv)
                     and _segment_present(vert_w, tr[0], tr[1], br[0], br[1], tv))
            if not sides:
                continue
            # one item fills the slot: no line across the middle
            if ky > 1 and any(_segment_present(horiz, x, y + k * g.pitch_y, tr[0], y + k * g.pitch_y, 0.6)
                              for k in range(1, ky)):
                continue
            if kx > 1 and any(_segment_present(vert, x + k * g.pitch_x, y, x + k * g.pitch_x, bl[1], 0.6)
                              for k in range(1, kx)):
                continue
            found.append(_snap_slot(g, horiz, vert, cell))
            break
    return found


def find_square_slots(bgr: np.ndarray, cell: float, box: tuple[int, int, int, int]) -> list[Grid]:
    """Single 1x1 slots inside an area (e.g. the attachment slots of an inspect
    window): squares of exactly one cell whose four sides are visible lines.
    The size tolerance is tight so a 'square' spanning the gap between two
    neighbouring slots is not mistaken for one."""
    bx, by, bw, bh = (int(v) for v in box)
    H, W = bgr.shape[:2]
    x0, y0, x1, y1 = max(0, bx), max(0, by), min(W, bx + bw), min(H, by + bh)
    if x1 - x0 < cell or y1 - y0 < cell:
        return []
    sub = bgr[y0:y1, x0:x1]
    horiz, vert = line_masks(sub, cell)
    cross = cv2.dilate(horiz, np.ones((3, 3), np.uint8)) & cv2.dilate(vert, np.ones((3, 3), np.uint8))
    _, _, _, cents = cv2.connectedComponentsWithStats(cross, connectivity=8)
    pts = cents[1:]
    if len(pts) < 4:
        return []
    hw = cv2.dilate(horiz, np.ones((3, 1), np.uint8))
    vw = cv2.dilate(vert, np.ones((1, 3), np.uint8))
    tol = 3.5

    def near(x, y):
        d = np.abs(pts[:, 0] - x) + np.abs(pts[:, 1] - y)
        k = int(np.argmin(d))
        return pts[k] if d[k] <= tol * 2 else None

    found: list[Grid] = []
    for x, y in sorted(map(tuple, pts), key=lambda p: (p[1], p[0])):
        tr, bl = near(x + cell, y), near(x, y + cell)
        if tr is None or bl is None:
            continue
        br = near(tr[0], bl[1])
        if br is None:
            continue
        if not (_segment_present(hw, x, y, tr[0], tr[1], 0.7) and _segment_present(hw, bl[0], bl[1], br[0], br[1], 0.7)
                and _segment_present(vw, x, y, bl[0], bl[1], 0.7) and _segment_present(vw, tr[0], tr[1], br[0], br[1], 0.7)):
            continue
        g = Grid(float(x + x0), float(y + y0), (tr[0] - x), (bl[1] - y), 1, 1, slot=True)
        if any(abs(o.x0 - g.x0) < cell * 0.5 and abs(o.y0 - g.y0) < cell * 0.5 for o in found):
            continue
        found.append(g)
    return found


def _snap_slot(g: Grid, horiz, vert, cell: float) -> Grid:
    """A slot's outer border can be a few px larger than the item inside it. Find
    the pair of lines exactly cols x rows cells apart (the item's own frame) that
    are strongest, and use that as the box."""
    x, y, w, h = g.rect
    H, W = vert.shape
    tw, th = int(round(g.cols * cell)), int(round(g.rows * cell))

    def col_score(cx):
        if cx < 0 or cx >= W:
            return 0.0
        return float(vert[max(0, y + 6):min(H, y + h - 6), cx].mean())

    def row_score(cy):
        if cy < 0 or cy >= H:
            return 0.0
        return float(horiz[cy, max(0, x + 6):min(W, x + w - 6)].mean())

    dx = max(range(-6, max(-5, w - tw + 7)), key=lambda d: (col_score(x + d) + col_score(x + d + tw), -abs(d)))
    dy = max(range(-6, max(-5, h - th + 7)), key=lambda d: (row_score(y + d) + row_score(y + d + th), -abs(d)))
    return Grid(g.x0 + dx, g.y0 + dy, tw / g.cols, th / g.rows, g.cols, g.rows, slot=True)


def _same_lattice(a: Grid, b: Grid, tol: float) -> bool:
    p = (a.pitch_x + a.pitch_y + b.pitch_x + b.pitch_y) / 4
    dx = (b.x0 - a.x0) / p
    dy = (b.y0 - a.y0) / p
    return abs(dx - round(dx)) * p <= tol and abs(dy - round(dy)) * p <= tol


def _gap(a: Grid, b: Grid) -> float:
    ax, ay, aw, ah = a.rect
    bx, by, bw, bh = b.rect
    gx = max(0, max(ax, bx) - min(ax + aw, bx + bw))
    gy = max(0, max(ay, by) - min(ay + ah, by + bh))
    return max(gx, gy)


def _union(a: Grid, b: Grid) -> Grid:
    p = (a.pitch_x * a.cols + b.pitch_x * b.cols) / (a.cols + b.cols)
    q = (a.pitch_y * a.rows + b.pitch_y * b.rows) / (a.rows + b.rows)
    x0, y0 = min(a.x0, b.x0), min(a.y0, b.y0)
    x1 = max(a.x0 + a.cols * a.pitch_x, b.x0 + b.cols * b.pitch_x)
    y1 = max(a.y0 + a.rows * a.pitch_y, b.y0 + b.rows * b.pitch_y)
    return Grid(x0, y0, p, q, max(1, round((x1 - x0) / p)), max(1, round((y1 - y0) / q)))


def _wall(mask: np.ndarray, g: Grid, a: float, b: float, horizontal: bool, tol: float) -> bool:
    """Is there a long line between coordinates a..b (exclusive of the lattice lines)?"""
    if horizontal:
        x0, x1 = int(g.x0), int(g.x0 + g.cols * g.pitch_x)
        for y in range(int(a) + 1, int(b)):
            off = (y - g.y0) / g.pitch_y
            if abs(off - round(off)) * g.pitch_y > tol + 2 and 0 <= y < mask.shape[0]:
                if mask[y, max(0, x0):x1].mean() >= 0.7:
                    return True
    else:
        y0, y1 = int(g.y0), int(g.y0 + g.rows * g.pitch_y)
        for x in range(int(a) + 1, int(b)):
            off = (x - g.x0) / g.pitch_x
            if abs(off - round(off)) * g.pitch_x > tol + 2 and 0 <= x < mask.shape[1]:
                if mask[max(0, y0):y1, x].mean() >= 0.7:
                    return True
    return False


def _trim_walls(g: Grid, masks, tol: float) -> Grid:
    """If a window border runs through a grid (e.g. the stash lattice leaked into the
    Quick Use bar below it), cut the grid there and keep the larger part."""
    if g.cols * g.rows < 8:
        return g
    for r in range(g.rows):
        y = g.y0 + r * g.pitch_y
        if _wall(masks[0], g, y, y + g.pitch_y, True, tol):
            if r + 1 > g.rows - r - 1:   # keep rows above (incl. a partly visible row r)
                return _trim_walls(Grid(g.x0, g.y0, g.pitch_x, g.pitch_y, g.cols, r), masks, tol) if r else g
            return _trim_walls(Grid(g.x0, y + g.pitch_y, g.pitch_x, g.pitch_y, g.cols, g.rows - r - 1), masks, tol)
    for c in range(g.cols):
        x = g.x0 + c * g.pitch_x
        if _wall(masks[1], g, x, x + g.pitch_x, False, tol):
            if c + 1 > g.cols - c - 1:
                return _trim_walls(Grid(g.x0, g.y0, g.pitch_x, g.pitch_y, c, g.rows), masks, tol) if c else g
            return _trim_walls(Grid(x + g.pitch_x, g.y0, g.pitch_x, g.pitch_y, g.cols - c - 1, g.rows), masks, tol)
    return g


def _divided(a: Grid, b: Grid, masks, tol: float) -> bool:
    """True if a window border runs between two grid fragments."""
    u = _union(a, b)
    top, bot = sorted([a, b], key=lambda g: g.y0)
    if bot.y0 >= top.y0 + top.rows * top.pitch_y - 2:
        if _wall(masks[0], u, top.y0 + top.rows * top.pitch_y, bot.y0, True, tol):
            return True
    left, right = sorted([a, b], key=lambda g: g.x0)
    if right.x0 >= left.x0 + left.cols * left.pitch_x - 2:
        if _wall(masks[1], u, left.x0 + left.cols * left.pitch_x, right.x0, False, tol):
            return True
    return False


def _merge_aligned(grids: list[Grid], pts: np.ndarray, cell: float, masks) -> list[Grid]:
    # Merging and growing feed each other (two fragments may only touch once both
    # have grown), so repeat until nothing changes.
    for _ in range(6):
        before = [(g.cols, g.rows, round(g.x0), round(g.y0)) for g in grids]
        grids = _merge_grow_once(grids, pts, cell, masks)
        if [(g.cols, g.rows, round(g.x0), round(g.y0)) for g in grids] == before:
            break
    return grids


def _merge_grow_once(grids: list[Grid], pts: np.ndarray, cell: float, masks) -> list[Grid]:
    """A stash packed with big items shows few grid lines, so it is found as many
    fragments. Fragments that sit on the same lattice and are close together are
    one grid; then the grid grows to any crossings (e.g. item corners) that are
    on its lattice just outside it."""
    tol = max(3.0, cell * 0.05)
    grids = list(grids)
    merged = True
    while merged:
        merged = False
        for i in range(len(grids)):
            for j in range(i + 1, len(grids)):
                a, b = grids[i], grids[j]
                big = max(a.cols * a.rows, b.cols * b.rows)
                if big < 4 or a.slot or b.slot:
                    continue  # never merge tiny slot grids (rig/pocket slots)
                if _same_lattice(a, b, tol) and _gap(a, b) <= 3 * cell and not _divided(a, b, masks, tol):
                    grids[i] = _union(a, b)
                    del grids[j]
                    merged = True
                    break
            if merged:
                break

    # Grow large grids to aligned crossings within 1.5 cells of their edge.
    out = []
    for g in grids:
        if g.cols * g.rows >= 8:
            # crossings that belong to other grids (e.g. the Quick Use bar) are off limits
            others = [o for o in grids if o is not g]
            free = pts[[not any(o.rect[0] - 4 <= x <= o.rect[0] + o.rect[2] + 4
                                and o.rect[1] - 4 <= y <= o.rect[1] + o.rect[3] + 4 for o in others)
                        for x, y in pts]] if others else pts
            for _ in range(10):
                x0, y0, w, h = g.rect
                near = free[(free[:, 0] >= x0 - 1.5 * cell) & (free[:, 0] <= x0 + w + 1.5 * cell)
                           & (free[:, 1] >= y0 - 1.5 * cell) & (free[:, 1] <= y0 + h + 1.5 * cell)]
                fx = (near[:, 0] - g.x0) / g.pitch_x
                fy = (near[:, 1] - g.y0) / g.pitch_y
                on = (np.abs(fx - np.round(fx)) * g.pitch_x <= tol) & (np.abs(fy - np.round(fy)) * g.pitch_y <= tol)
                near = near[on]
                if not len(near):
                    break
                c0 = min(0, int(np.round((near[:, 0].min() - g.x0) / g.pitch_x)))
                r0 = min(0, int(np.round((near[:, 1].min() - g.y0) / g.pitch_y)))
                c1 = max(g.cols, int(np.round((near[:, 0].max() - g.x0) / g.pitch_x)))
                r1 = max(g.rows, int(np.round((near[:, 1].max() - g.y0) / g.pitch_y)))
                # never grow across a window border (a long line that is off the lattice)
                if r1 > g.rows and _wall(masks[0], g, g.y0 + g.rows * g.pitch_y, g.y0 + r1 * g.pitch_y, True, tol):
                    r1 = g.rows
                if r0 < 0 and _wall(masks[0], g, g.y0 + r0 * g.pitch_y, g.y0, True, tol):
                    r0 = 0
                if c1 > g.cols and _wall(masks[1], g, g.x0 + g.cols * g.pitch_x, g.x0 + c1 * g.pitch_x, False, tol):
                    c1 = g.cols
                if c0 < 0 and _wall(masks[1], g, g.x0 + c0 * g.pitch_x, g.x0, False, tol):
                    c0 = 0
                if (c0, r0, c1, r1) == (0, 0, g.cols, g.rows):
                    break
                g = Grid(g.x0 + c0 * g.pitch_x, g.y0 + r0 * g.pitch_y, g.pitch_x, g.pitch_y, c1 - c0, r1 - r0)
        out.append(g)
    # Drop small grids swallowed by a grown grid.
    return [g for g in out if not any(o is not g and o.cols * o.rows > g.cols * g.rows
                                      and o.contains(g.x0 + 2, g.y0 + 2)
                                      and o.contains(g.x0 + g.cols * g.pitch_x - 2, g.y0 + g.rows * g.pitch_y - 2)
                                      for o in out)]


def _edge_is_frame(gray: np.ndarray, x1, y1, x2, y2) -> bool:
    """True if the border between two points is an item frame: a thin line that is
    bright AND brighter than the pixels a few px either side of it. Bright item art
    (e.g. an MRE packet) crossing a grid line is bright but has no such ridge."""
    x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
    h, w = gray.shape
    if x1 == x2:  # vertical border: profiles run horizontally
        pad = max(4, (y2 - y1) // 10)
        if x1 - 5 < 0 or x1 + 6 > w:
            return False
        prof = gray[max(0, y1 + pad):min(h, y2 - pad), x1 - 5:x1 + 6]
    else:
        pad = max(4, (x2 - x1) // 10)
        if y1 - 5 < 0 or y1 + 6 > h:
            return False
        prof = gray[y1 - 5:y1 + 6, max(0, x1 + pad):min(w, x2 - pad)].T
    if prof.size == 0:
        return False
    prof = prof.astype(np.int16)                     # (samples, 11)
    centre = prof[:, 2:9]                             # allow +-3 px misalignment
    i = centre.argmax(axis=1) + 2
    rows = np.arange(len(prof))
    peak = prof[rows, i]
    sides = np.maximum(prof[rows, np.clip(i - 3, 0, 10)], prof[rows, np.clip(i + 3, 0, 10)])
    if np.median(peak) >= BORDER_BRIGHT and np.median(peak - sides) >= BORDER_RIDGE:
        return True
    # On lighter screens (e.g. the blurred in-raid menu backdrop) tile backgrounds
    # are brighter, so the frame stands out less. It is still one constant grey
    # line, which item art crossing a grid line never is.
    for j in range(3, 8):
        col = prof[:, j]
        med = np.median(col)
        if (np.mean(np.abs(col - med) <= 6) >= 0.85 and med >= BORDER_BRIGHT
                and med - np.median(prof[:, max(0, j - 3)]) >= 8 and med - np.median(prof[:, min(10, j + 3)]) >= 8):
            return True
    # A highlighted tile (mouse hover, tracked item) has a light background, so its
    # frame is no longer brighter than both sides. Its border is still a clean step
    # between two flat backgrounds; item art crossing a grid line is not flat.
    side_a = prof[:, 0:2].mean(axis=1)
    side_b = prof[:, 9:11].mean(axis=1)
    step = side_b - side_a
    if len(step) < 6 or abs(np.median(step)) < BORDER_STEP:
        return False
    agree = np.mean(np.sign(step) == np.sign(np.median(step)))
    # A frame is a thin line of one constant grey along the whole border. Item art
    # touching the border (counters, magazines) can blur the step, not the line.
    uniform_line = any(np.mean(np.abs(prof[:, j] - np.median(prof[:, j])) <= 8) >= 0.8 for j in range(3, 8))
    if uniform_line and agree >= 0.65 and abs(np.median(step)) >= 30:
        return True
    consistent = agree >= 0.85
    # flat backgrounds: most samples near the typical value (art may touch the edge here and there)
    flat_a = np.mean(np.abs(side_a - np.median(side_a)) <= 12) >= 0.75
    flat_b = np.mean(np.abs(side_b - np.median(side_b)) <= 12) >= 0.75
    if flat_a and flat_b:
        return bool(consistent)
    # a big, consistent jump with at least one flat side (art may fill the other)
    return bool(consistent and (flat_a or flat_b) and abs(np.median(step)) >= 25)


def _edge_visible_frame(gray: np.ndarray, x1, y1, x2, y2, occluders) -> bool:
    """_edge_is_frame, judged only on the part of the border not hidden by an
    occluder (e.g. a hover tooltip). A border that is mostly hidden counts as
    'no frame', so the cells either side are treated as one item."""
    if not occluders:
        return _edge_is_frame(gray, x1, y1, x2, y2)
    vertical = abs(x2 - x1) < 1
    a0, a1 = (y1, y2) if vertical else (x1, x2)
    pos = x1 if vertical else y1
    hidden = []
    for ox, oy, ow, oh in occluders:
        if vertical and ox - 6 <= pos <= ox + ow + 6:
            hidden.append((oy, oy + oh))
        elif not vertical and oy - 6 <= pos <= oy + oh + 6:
            hidden.append((ox, ox + ow))
    # longest visible stretch of the border
    cuts = sorted((max(a0, s), min(a1, e)) for s, e in hidden if e > a0 and s < a1)
    best, cur = (a0, a0), a0
    for s, e in cuts:
        if s - cur > best[1] - best[0]:
            best = (cur, s)
        cur = max(cur, e)
    if a1 - cur > best[1] - best[0]:
        best = (cur, a1)
    if not cuts:
        return _edge_is_frame(gray, x1, y1, x2, y2)
    if best[1] - best[0] < 0.3 * (a1 - a0):
        return None  # hidden: decided later from the neighbouring borders on the same line
    if vertical:
        return _edge_is_frame(gray, x1, best[0], x2, best[1])
    return _edge_is_frame(gray, best[0], y1, best[1], y2)


def _hatched(gray: np.ndarray, rect) -> bool:
    """Cells of a container that hasn't been searched yet are drawn with diagonal
    stripes: nearly all strong edges are diagonal and all slope the same way."""
    x, y, w, h = rect
    patch = gray[y + 8:y + h - 8, x + 8:x + w - 8].astype(np.float32)
    if patch.size < 100:
        return False
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    strong = mag > np.percentile(mag, 70)
    diagonal = (np.abs(np.abs(gx) - np.abs(gy)) < 0.4 * mag) & strong
    if strong.sum() == 0 or diagonal.sum() / strong.sum() < 0.8:
        return False
    return abs(np.sign(gx * gy)[diagonal].mean()) >= 0.95


def _fill_hidden(edges: np.ndarray, along_rows: bool) -> None:
    """Borders hidden by an occluder copy the nearest visible borders on the same grid
    line: a frame running along both sides of a tooltip continues underneath it."""
    lines = edges if along_rows else edges.T
    for line in lines:
        for i, v in enumerate(line):
            if v is not None:
                continue
            left = next((line[j] for j in range(i - 1, -1, -1) if line[j] is not None), None)
            right = next((line[j] for j in range(i + 1, len(line)) if line[j] is not None), None)
            known = [b for b in (left, right) if b is not None]
            line[i] = bool(known) and all(known)


def _texture(gray: np.ndarray, rect) -> float:
    """Mean edge strength inside a region. Empty slots are a flat, see-through
    background (about 3-6); any item - even a weapon on its near-black background -
    has an icon and a coloured tile (about 18+)."""
    x, y, w, h = rect
    ix, iy = int(w * 0.12), int(h * 0.12)
    patch = gray[y + iy:y + h - iy, x + ix:x + w - ix].astype(np.float32)
    if patch.size == 0:
        return 0.0
    grad = np.abs(cv2.Sobel(patch, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(patch, cv2.CV_32F, 0, 1))
    return float(grad.mean())


def split_items(bgr: np.ndarray, grids: list[Grid], occluders=(), forced=(), cell: float = 0.0) -> list[Region]:
    """Split every grid into rectangular candidate regions (items or empty space)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    regions: list[Region] = []
    for gi, g in enumerate(grids):
        # An occluder (tooltip, inspect window) hides grids beneath it, not grids inside it.
        gcx, gcy = g.x0 + g.cols * g.pitch_x / 2, g.y0 + g.rows * g.pitch_y / 2
        occluders_g = [o for o in occluders if not (o[0] <= gcx <= o[0] + o[2] and o[1] <= gcy <= o[1] + o[3])]
        # is every cell border (including the grid's outer edge) an item frame?
        # None = hidden under an occluder, filled in from its neighbours below.
        v_strong = np.zeros((g.rows, g.cols + 1), object)  # frame on the left of cell c
        h_strong = np.zeros((g.rows + 1, g.cols), object)  # frame above cell r
        for r in range(g.rows):
            for c in range(g.cols + 1):
                x = g.x0 + c * g.pitch_x
                v_strong[r, c] = _edge_visible_frame(gray, x, g.y0 + r * g.pitch_y, x, g.y0 + (r + 1) * g.pitch_y, occluders_g)
        for r in range(g.rows + 1):
            for c in range(g.cols):
                y = g.y0 + r * g.pitch_y
                h_strong[r, c] = _edge_visible_frame(gray, g.x0 + c * g.pitch_x, y, g.x0 + (c + 1) * g.pitch_x, y, occluders_g)
        _fill_hidden(h_strong, along_rows=True)
        _fill_hidden(v_strong, along_rows=False)
        v_strong, h_strong = v_strong.astype(bool), h_strong.astype(bool)
        if cell and (abs(g.pitch_x - cell) > 0.04 * cell or abs(g.pitch_y - cell) > 0.04 * cell):
            # Spacing wider than a cell: an array of separate 1x1 slots with gaps
            # between them (rig pouches). Each slot holds its own item.
            v_strong[:, :] = True
            h_strong[:, :] = True

        # Known item placements (e.g. the hovered item, whose highlight hides its frame).
        for fgi, (fx, fy, fw, fh) in forced:
            if fgi != gi:
                continue
            c0, r0 = int(round((fx - g.x0) / g.pitch_x)), int(round((fy - g.y0) / g.pitch_y))
            c1, r1 = c0 + int(round(fw / g.pitch_x)), r0 + int(round(fh / g.pitch_y))
            for r in range(r0, r1):
                v_strong[r, c0] = v_strong[r, c1] = True
                v_strong[r, c0 + 1:c1] = False
            for c in range(c0, c1):
                h_strong[r0, c] = h_strong[r1, c] = True
                h_strong[r0 + 1:r1, c] = False

        idx = lambda c, r: r * g.cols + c  # noqa: E731
        uf = _UF(g.cols * g.rows)
        for r in range(g.rows):
            for c in range(g.cols):
                if c + 1 < g.cols and not v_strong[r, c + 1]:
                    uf.union(idx(c, r), idx(c + 1, r))
                if r + 1 < g.rows and not h_strong[r + 1, c]:
                    uf.union(idx(c, r), idx(c, r + 1))

        comps: dict[int, list[tuple[int, int]]] = {}
        for r in range(g.rows):
            for c in range(g.cols):
                comps.setdefault(uf.find(idx(c, r)), []).append((c, r))

        found: list[Region] = []
        for cells in comps.values():
            cs = [c for c, _ in cells]
            rs = [r for _, r in cells]
            c0, r0 = min(cs), min(rs)
            w, h = max(cs) - c0 + 1, max(rs) - r0 + 1
            if w * h != len(cells):
                continue  # not a rectangle -> empty space between items
            outline = ([v_strong[r, c0] for r in range(r0, r0 + h)]
                       + [v_strong[r, c0 + w] for r in range(r0, r0 + h)]
                       + [h_strong[r0, c] for c in range(c0, c0 + w)]
                       + [h_strong[r0 + h, c] for c in range(c0, c0 + w)])
            x, y, _, _ = g.cell_rect(c0, r0)
            rect = (x, y, int(round(w * g.pitch_x)), int(round(h * g.pitch_y)))
            found.append(Region(gi, c0, r0, w, h, rect, framed=float(np.mean(outline))))

        for reg in found:
            reg.looks_empty = _texture(gray, reg.rect) < EMPTY_TEXTURE
            reg.unsearched = not reg.looks_empty and _hatched(gray, reg.rect)
        regions.extend(found)
    return regions
