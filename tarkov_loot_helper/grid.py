"""Detects inventory grids (stash, backpack, rig, pockets, containers...) and
splits them into item regions.

Approach:
  1. Find thin grid lines that are brighter than their surroundings.
  2. Keep only long horizontal / vertical runs (at least ~60% of a cell).
  3. Grid-line crossings become lattice points; points one cell apart that are
     joined by a visible line are linked. Each connected lattice is one grid.
  4. Inside a grid, cell borders that are *missing* mean an item image covers
     them, so neighbouring cells are merged into one multi-cell item.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Grid:
    x0: float
    y0: float
    pitch_x: float
    pitch_y: float
    cols: int
    rows: int

    def cell_rect(self, c: int, r: int) -> tuple[int, int, int, int]:
        x = self.x0 + c * self.pitch_x
        y = self.y0 + r * self.pitch_y
        return int(round(x)), int(round(y)), int(round(self.pitch_x)), int(round(self.pitch_y))

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (int(self.x0), int(self.y0),
                int(round(self.cols * self.pitch_x)), int(round(self.rows * self.pitch_y)))


@dataclass
class Region:
    grid_index: int
    col: int
    row: int
    width: int   # in cells
    height: int  # in cells
    rect: tuple[int, int, int, int]  # x, y, w, h in image pixels


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
    mask = (highpass > 5).astype(np.uint8)
    k = max(8, int(cell * 0.6))
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, k), np.uint8))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, 1), np.uint8))
    # Tolerate 1px jitter when sampling segments later.
    horiz = cv2.dilate(horiz, np.ones((3, 1), np.uint8))
    vert = cv2.dilate(vert, np.ones((1, 3), np.uint8))
    return horiz, vert


def _segment_present(mask: np.ndarray, x1, y1, x2, y2, thresh=0.5) -> bool:
    n = int(max(abs(x2 - x1), abs(y2 - y1)))
    if n < 4:
        return False
    # Skip the ends so the crossing itself doesn't count.
    t = np.linspace(0.15, 0.85, max(4, n // 2))
    xs = np.clip(np.round(x1 + (x2 - x1) * t).astype(int), 0, mask.shape[1] - 1)
    ys = np.clip(np.round(y1 + (y2 - y1) * t).astype(int), 0, mask.shape[0] - 1)
    return mask[ys, xs].mean() >= thresh


def detect_grids(bgr: np.ndarray, cell: float) -> tuple[list[Grid], tuple[np.ndarray, np.ndarray]]:
    horiz, vert = line_masks(bgr, cell)
    cross = cv2.dilate(horiz, np.ones((3, 3), np.uint8)) & cv2.dilate(vert, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(cross, connectivity=8)
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
        minx, maxx = g[:, 0].min(), g[:, 0].max()
        miny, maxy = g[:, 1].min(), g[:, 1].max()
        cols = int(round((maxx - minx) / cell))
        rows = int(round((maxy - miny) / cell))
        if cols < 1 or rows < 1:
            continue
        grids.append(Grid(minx, miny, (maxx - minx) / cols, (maxy - miny) / rows, cols, rows))
    grids.sort(key=lambda gr: (gr.y0, gr.x0))
    return grids, (horiz, vert)


def _cell_stats(bgr: np.ndarray, rect) -> tuple[float, float]:
    x, y, w, h = rect
    ix, iy = int(w * 0.2), int(h * 0.2)
    patch = bgr[y + iy:y + h - iy, x + ix:x + w - ix]
    if patch.size == 0:
        return 0.0, 0.0
    return float(patch.mean()), float(patch.std())


def split_items(bgr: np.ndarray, grids: list[Grid], masks) -> list[Region]:
    horiz, vert = masks
    regions: list[Region] = []
    for gi, g in enumerate(grids):
        idx = lambda c, r: r * g.cols + c  # noqa: E731
        uf = _UF(g.cols * g.rows)
        for r in range(g.rows):
            for c in range(g.cols):
                x, y = g.x0 + c * g.pitch_x, g.y0 + r * g.pitch_y
                if c + 1 < g.cols:  # border to the right
                    bx = x + g.pitch_x
                    if not _segment_present(vert, bx, y, bx, y + g.pitch_y):
                        uf.union(idx(c, r), idx(c + 1, r))
                if r + 1 < g.rows:  # border below
                    by = y + g.pitch_y
                    if not _segment_present(horiz, x, by, x + g.pitch_x, by):
                        uf.union(idx(c, r), idx(c, r + 1))

        comps: dict[int, list[tuple[int, int]]] = {}
        for r in range(g.rows):
            for c in range(g.cols):
                comps.setdefault(uf.find(idx(c, r)), []).append((c, r))

        # Baseline for an empty slot: the median of all isolated cells (most are empty).
        singles = {k: _cell_stats(bgr, g.cell_rect(*v[0])) for k, v in comps.items() if len(v) == 1}
        empty_mean = float(np.median([m for m, _ in singles.values()])) if singles else 0.0

        for key, cells in comps.items():
            cs = [c for c, _ in cells]
            rs = [r for _, r in cells]
            c0, r0 = min(cs), min(rs)
            w, h = max(cs) - c0 + 1, max(rs) - r0 + 1
            if w * h != len(cells):
                continue  # not a rectangle -> unreliable, skip
            if len(cells) == 1:
                mean, std = singles[key]
                if std < 12.0 and abs(mean - empty_mean) < 12.0:
                    continue  # empty slot
            x, y, _, _ = g.cell_rect(c0, r0)
            rect = (x, y, int(round(w * g.pitch_x)), int(round(h * g.pitch_y)))
            regions.append(Region(gi, c0, r0, w, h, rect))
    return regions
