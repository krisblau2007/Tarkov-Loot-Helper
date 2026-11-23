# Tarkov Loot Helper

Screen-based inventory scanner for Escape From Tarkov. Press a hotkey with any
inventory open (stash, backpack, rig, pockets, containers, loot screens) and an
overlay labels each recognised item with its value per slot and marks items
needed for quests or the hideout.

It only takes screenshots. It never reads or modifies game memory or files.

## Quick start

Double-click **`Launch Tarkov Loot Helper.bat`**. On the first run it sets
everything up, then it starts the app in the system tray.

## Manual setup

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
   item, yellow = equipped item, red = low-confidence guess.
3. Adjust UI scale or *Cell size @1080p* in Settings. You can also re-test a
   saved screenshot offline:
   `python -m tarkov_loot_helper scan debug/XXXX_raw.png --ui-scale 1.0`

Items whose IDs are listed under `blacklist` in `settings.json` are never shown
(the WI-FI Camera in the special slots is blacklisted by default).

## Project layout

- `tarkov_loot_helper/Application`: tray app, tarkov.dev API connector, item model
- `tarkov_loot_helper/Configuration`: settings and the settings dialog
- `tarkov_loot_helper/Identification`: screen capture, grid detection, icon matching, overlay, export

## How it works

1. `Identification/grid.py`: finds thin, bright grid lines, links the points where they
   cross into lattices (one per container), then merges cells whose shared
   border is hidden by an item image into multi-cell items.
2. `Identification/identify.py`: works out each tile's footprint (cols x rows)
   first, then reads its short name with Windows OCR (`ocr.py`) and matches it to
   items of that size. Guns are matched by name and caliber at any size and are
   valued as the base weapon. Icon comparison breaks ties and is the fallback.
   `equipment.py` does the same for equipped items in the gear slot panels.
3. `Application/TarkovDevAPIConnector.py`: all requests to the tarkov.dev JSON API
   (`https://json.tarkov.dev/{gameMode}/items`, `/tasks`, `/hideout`,
   `/traders`, plus `_{lang}` translations), cached for an hour in `data/`.
   The game modes come from `/endpoints` and are saved in `settings.json`.
   `Application/api.py` turns that raw data into `Item` objects.
