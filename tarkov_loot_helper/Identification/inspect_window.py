"""Item inspect windows (opened from an item's context menu).

They show the item's attachment / armor slots as a row of tiles. Rules:
  * Empty slots are labelled with the slot type in capitals ("MOUNT", "SCOPE",
    "SHROUD"...): not items.
  * Built-in armor parts of helmets/armor ("TOP", "NAPE"...) never change: ignored.
  * Attachments that carry other attachments (rails, mounts, handguards, dust
    covers with rails) are ignored: what is mounted on them can't be seen.
"""
from __future__ import annotations

from tarkov_loot_helper.Application.api import Item
from tarkov_loot_helper.Identification.ocr import OcrLine
from tarkov_loot_helper.Identification.textmatch import normalise

BUTTONS = {normalise(b) for b in (
    "RELOAD", "UNLOAD", "REPAIR", "INSURE", "UNEQUIP", "UNINSTALL", "FOLD", "UNFOLD", "EDIT BUILD",
    "DISASSEMBLE", "FILTER BY ITEM", "LINKED SEARCH", "REQUIRED SEARCH", "ADD TO WISHLIST", "DISCARD",
    "EQUIP", "INSTALL", "USE")}

# Labels Tarkov prints on empty attachment/armor slots (and built-in armor parts).
SLOT_LABELS = [normalise(s) for s in (
    "MOUNT", "SCOPE", "SIGHT", "REAR SIGHT", "FRONT SIGHT", "TACTICAL", "MUZZLE", "BARREL", "HANDGUARD",
    "STOCK", "PISTOL GRIP", "MAGAZINE", "CHARGE", "RECEIVER", "GAS BLOCK", "FOREGRIP", "SHROUD", "TOP",
    "NAPE", "EARS", "EYES", "JAW", "FRONT", "BACK", "LEFT SIDE", "RIGHT SIDE", "GROIN", "COLLAR",
    "BUTTPAD", "LAUNCHER", "CATCH", "HAMMER", "TRIGGER", "CHAMBER", "PATRON", "BIPOD", "SOFT ARMOR",
    "FRONT PLATE", "BACK PLATE", "LEFT SIDE PLATE", "RIGHT SIDE PLATE", "HELMET", "EQUIPMENT", "MOD")]


def find_windows(lines: list[OcrLine], cell: float) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of inspect windows, found from their row of action buttons."""
    wish = normalise("ADD TO WISHLIST")
    boxes = []
    for anchor in lines:
        if normalise(anchor.text) != wish:
            continue
        ax, ay, aw, ah = anchor.rect
        row = [l for l in lines if normalise(l.text) in BUTTONS
               and abs(l.rect[1] - ay) <= 1.6 * cell and abs(l.rect[0] - ax) <= 10 * cell]
        left = min(l.rect[0] for l in row)
        right = max(l.rect[0] + l.rect[2] for l in row)
        x0 = int(left - 1.2 * cell)  # button icons + window padding left of the labels
        top = int(ay - 6.7 * cell)  # the window title sits about 6 cells above the buttons
        boxes.append((x0, top, int(right + 0.9 * cell - x0), int(15.2 * cell)))
    return boxes


def in_title_band(box, x: float, y: float, cell: float) -> bool:
    """The window's title (the item's full name) - not a hover tooltip."""
    bx, by, bw, _ = box
    return bx - cell <= x <= bx + bw and by <= y <= by + 1.6 * cell


def inside(box, x: float, y: float) -> bool:
    bx, by, bw, bh = box
    return bx <= x <= bx + bw and by <= y <= by + bh


def is_slot_label(text: str) -> bool:
    """True for an empty-slot label, incl. truncated ones ("FRONT SIG...")."""
    t = text.strip()
    if not t or t.upper() != t:  # slot labels are all capitals
        return False
    key = normalise(t)
    return len(key) >= 3 and any(s == key or (s.startswith(key) and len(key) >= 4) for s in SLOT_LABELS)


def is_empty_slot_tile(tile_bgr) -> bool:
    """Empty attachment slots are a grey tile with a grey silhouette; installed
    parts sit on a tinted tile (saturation ~55+ vs ~10)."""
    import cv2
    h, w = tile_bgr.shape[:2]
    body = tile_bgr[int(h * 0.26):int(h * 0.83), int(w * 0.1):int(w * 0.9)]
    if body.size == 0:
        return False
    return float(cv2.cvtColor(body, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) < 28


def ignored_in_window(item: Item) -> bool:
    """Attachments that hold other attachments, and built-in armor parts."""
    return item.has_slots or item.armor_part
