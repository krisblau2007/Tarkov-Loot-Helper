"""Identifies an item region by comparing it against tarkov.dev grid icons of
the same footprint (width x height, or rotated)."""
from __future__ import annotations

import cv2
import numpy as np

from tarkov_loot_helper.Application.api import Item
from tarkov_loot_helper.Configuration.config import DATA_DIR
from tarkov_loot_helper.Identification.icons import icon_path, load_icon
from tarkov_loot_helper.Identification.textmatch import caliber_similarity, name_similarity, normalise

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


def _zvec(a: np.ndarray) -> np.ndarray:
    a = a.reshape(-1).astype(np.float32)
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def _features(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Feature vector whose dot product with another is 0.5*colour correlation +
    0.5*shape correlation. Shape (edge strength) ignores the tile's background
    colour, which differs in game (e.g. green magazine tiles, player-set colours)."""
    norm = _normalise(img, w, h)
    mask = _mask(w, h)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    edges = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    return np.concatenate([_zvec(norm[mask]), _zvec(edges[mask])]) * np.float32(np.sqrt(0.5))


def _hue_hist(img: np.ndarray) -> np.ndarray:
    """Hue/saturation histogram of the item itself: the central area, ignoring
    dark and grey pixels (background, frame, text)."""
    h, w = img.shape[:2]
    core = img[int(h * 0.2):int(h * 0.85), int(w * 0.1):int(w * 0.9)]
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 45) & (hsv[:, :, 1] > 40)).astype(np.uint8)
    hist = cv2.calcHist([hsv], [0, 1], mask, [18, 4], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


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
            if it.is_preset or it.is_gun:  # guns are identified by name + caliber
                continue
            for rotated in (False, True):
                iw, ih = (it.height, it.width) if rotated else (it.width, it.height)
                if (iw, ih) != (w, h) or (rotated and it.width == it.height):
                    continue
                candidates.append((it, rotated))

        # Reuse the prepared templates from disk if the item set hasn't changed.
        cache_file = TEMPLATE_DIR / f"{w}x{h}_s{S}_v2.npz"
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
            imgs.append(_features(icon, w, h))
        mask = _mask(w, h)
        if imgs:
            arr = np.stack(imgs)
        else:
            arr = np.zeros((0, int(mask.sum()) * 4), np.float32)
        self._bank[key] = (refs, arr)
        if refs:
            TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
            keys = np.array([f"{it.id}:{int(rot)}" for it, rot in refs])
            np.savez(cache_file, keys=keys, arr=arr.astype(np.float32))
        return self._bank[key]

    def icon_scores(self, crop: np.ndarray, w: int, h: int) -> dict[str, tuple[float, bool]]:
        """item id -> (correlation with its icon, rotated) for every item of this footprint."""
        refs, bank = self._templates(w, h)
        if not refs:
            return {}
        scores = bank @ _features(crop, w, h)  # colour + shape correlation per template
        out: dict[str, tuple[float, bool]] = {}
        for (it, rot), s in zip(refs, scores):
            if it.id not in out or s > out[it.id][0]:
                out[it.id] = (float(s), rot)
        return out

    def identify(self, crop: np.ndarray, w: int, h: int) -> tuple[Item | None, float, bool]:
        """Icon-only identification (fallback when no text was read)."""
        scores = self.icon_scores(crop, w, h)
        if not scores:
            return None, 0.0, False
        iid = max(scores, key=lambda k: scores[k][0])
        return self.items[iid], scores[iid][0], scores[iid][1]

    def _colour_match(self, target: np.ndarray, item: Item) -> float:
        if not hasattr(self, "_hue_cache"):
            self._hue_cache: dict[str, np.ndarray | None] = {}
        if item.id not in self._hue_cache:
            icon = load_icon(item)
            self._hue_cache[item.id] = None if icon is None else _hue_hist(icon)
        ref = self._hue_cache[item.id]
        return -1.0 if ref is None else float(cv2.compareHist(target, ref, cv2.HISTCMP_CORREL))

    def _guns(self) -> list[Item]:
        if not hasattr(self, "_gun_list"):
            self._gun_list = [it for it in self.items.values() if it.is_gun and not it.is_preset]
        return self._gun_list

    def identify_region(self, crop: np.ndarray, w: int, h: int, name_text: str,
                        caliber_text: str = "", occluded: bool = False,
                        capacity: int = 0, extend: bool = False) -> tuple[Item | None, float, str]:
        """Returns (item, confidence 0..1, method) where method is 'name', 'gun' or 'icon'.
        `capacity` is the magazine size read from the tile's "x/17" counter (0 = unknown)."""
        icons = self.icon_scores(crop, w, h)
        best: tuple[float, Item | None, str] = (0.0, None, "")
        if name_text:
            named = []
            for iid, (icon_s, _) in icons.items():
                it = self.items[iid]
                ns = name_similarity(name_text, it.short_name)
                if ns >= 0.6:  # footprint already matches; the icon helps confirm
                    named.append((ns, icon_s, it))
            mag_confirmed = bool(capacity) and any(it.capacity == capacity for _, _, it in named)
            if mag_confirmed:
                # Magazines with truncated names ("Glock 9x1"): the counter settles it.
                named = [(n + (0.05 if it.capacity == capacity else 0.0), s, it) for n, s, it in named
                         if not it.capacity or it.capacity == capacity]
            if extend:
                # The tile is cut off by a tooltip/window: the item may be bigger than
                # what is visible. Same-named items with a larger footprint qualify too.
                top_ns = max((n for n, _, _ in named), default=0.0)
                for it in self.items.values():
                    if it.is_preset or it.is_gun:
                        continue
                    fits = (it.width >= w and it.height >= h) or (it.height >= w and it.width >= h)
                    if fits and it.width * it.height > w * h:
                        ns = name_similarity(name_text, it.short_name)
                        if ns >= 0.9 and ns >= top_ns - 0.02:
                            named.append((ns + 0.01, 0.0, it))  # prefer the bigger, uncut item
            if named:
                top_ns = max(n for n, _, _ in named)
                tied = [(n, s, it) for n, s, it in named if n >= top_ns - 0.02]
                if len(tied) > 1 and occluded:
                    # Partly hidden (e.g. under a tooltip): colour can't be judged, so take
                    # the standard variant (shortest full name, no colour suffix).
                    shortest = min(len(it.name) for _, _, it in tied)
                    tied = [(n, 1.0 if len(it.name) == shortest else 0.0, it) for n, _, it in tied]
                elif len(tied) > 1:
                    # Same name, different variant (e.g. black vs Plum 6L23): colour decides.
                    target = _hue_hist(crop)
                    tied = [(n, s + 0.5 * self._colour_match(target, it), it) for n, s, it in tied]
                n, s, it = max(tied, key=lambda t: t[1])
                best = (min(1.0, 0.8 * n + 0.2 * max(0.0, min(1.0, s))), it, "name")
            # Guns: any footprint of 2+ cells, match by name, confirm by caliber.
            # (Not when a magazine counter already matched a magazine: "AK-12" mag.)
            for it in self._guns() if w * h >= 2 and not mag_confirmed else ():
                ns = name_similarity(name_text, it.short_name)
                if ns < 0.75:
                    continue
                cs = caliber_similarity(caliber_text, it.caliber) if caliber_text else 0.5
                score = 0.75 * ns + 0.25 * cs
                if w * h > 1:
                    score += 0.05  # a multi-cell tile is more likely a gun than a 1x1 part
                if score > best[0]:
                    best = (min(score, 1.0), it, "gun")
            # Name didn't fit the footprint: special slots and other fixed slots show
            # items shrunk to one cell (e.g. a 2x1 CMS kit), so search every item.
            if best[0] < 0.8 and len(normalise(name_text)) >= 3:
                for it in self.items.values():
                    if it.is_preset or it.is_gun:
                        continue
                    ns = name_similarity(name_text, it.short_name)
                    if ns >= 0.9 and ns * 0.85 > best[0]:
                        best = (ns * 0.85, it, "name")
        icon_best = max(icons, key=lambda k: icons[k][0]) if icons else None
        icon_score = icons[icon_best][0] if icon_best else 0.0
        # A near-perfect icon match beats a weak name guess. Example: dogtags show the
        # player's nickname instead of an item name.
        if icon_best and icon_score >= 0.85 and icon_score > best[0]:
            return self.items[icon_best], icon_score, "icon"
        # Dogtags show a player's nickname, which can fuzzily resemble a real item name
        # ("Vintorezz"). A clear dogtag icon wins over anything but a near-exact name.
        if (icon_best and "Dogtag" in self.items[icon_best].name and icon_score >= 0.6
                and best[0] < 0.9):
            return self.items[icon_best], max(icon_score, 0.8), "icon"
        if best[1] is not None:
            return best[1], best[0], best[2]
        if capacity:
            # No readable name (e.g. "DD"), but a magazine counter: only magazines of that
            # size and footprint qualify, which usually leaves very few candidates.
            sized = {k: v for k, v in icons.items() if self.items[k].capacity == capacity}
            if sized:
                ranked = sorted(sized, key=lambda k: sized[k][0], reverse=True)
                k = ranked[0]
                score = sized[k][0]
                runner_up = sized[ranked[1]][0] if len(ranked) > 1 else -1.0
                if len(ranked) == 1 or score - runner_up >= 0.15:
                    score = max(score, 0.75)  # the only (or clearly best) mag of that size
                return self.items[k], score, "icon"
        if icon_best:
            return self.items[icon_best], icon_score, "icon"
        return None, 0.0, ""
