# Tarkov Loot Helper

Screen-based inventory scanner for Escape From Tarkov. Press a hotkey with any
inventory open (stash, backpack, rig, pockets, containers, loot screens) and an
overlay labels each recognised item with its value per slot and marks items
needed for quests or the hideout.

It only takes screenshots. It never reads or modifies game memory or files.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m tarkov_loot_helper setup   # item data + ~4k icons (one-time)
```

## Use

```powershell
.\.venv\Scripts\python -m tarkov_loot_helper
```

Run Tarkov in **Borderless Windowed** mode so the overlay can draw on top of it.

| Key | Action |
|-----|--------|
| F8  | Scan the screen and show the overlay |
| F9  | Hide the overlay |

Overlay colours: **yellow** = quest item (Q), **blue** = hideout item (H),
**green** = above the valuable threshold, grey = other.

Open the tray icon menu to change the **UI scale** (quick presets or exact
value under Settings), export the last scan to CSV/JSON (`exports/`), refresh
prices, switch between PvP and PvE prices, or save debug images.

## Tuning

If grids aren't found, the cell size is probably wrong for your display:
1. Tick **Save debug images** in the tray menu and scan once.
2. Check `debug/*_annotated.png`: orange = detected grid, green = identified
   item, red = unidentified item.
3. Adjust UI scale or *Cell size @1080p* in Settings. You can also re-test a
   saved screenshot offline:
   `python -m tarkov_loot_helper scan debug/XXXX_raw.png --ui-scale 1.0`

## How it works

1. `grid.py`: finds thin, bright grid lines, links the points where they
   cross into lattices (one per container), then merges cells whose shared
   border is hidden by an item image into multi-cell items.
2. `identify.py`: compares each item region with tarkov.dev icons of the
   same footprint (including rotated items), ignoring the name and stack-count
   corners.
3. `api.py`: prices, quest and hideout requirements from api.tarkov.dev,
   cached for an hour in `data/`.
